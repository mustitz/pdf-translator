import os

from argparse import ArgumentParser
from functools import lru_cache, partial

import cv2
import fitz
import yaml

from easyocr import Reader as EasyocrReader
from torch import cuda

dump_yaml = yaml.dump
load_yaml = partial(yaml.load, Loader=yaml.FullLoader)


@lru_cache
def get_easyocr_reader(lang_from, *, gpu=None, verbose=False):
    if lang_from != 'en':
        langs = ['en', lang_from]
    else:
        langs = ['en']

    langs_str = '+'.join(langs)
    gpu_str = 'auto' if gpu is None else str(gpu).lower()
    verbose_str = str(verbose).lower()

    print(f"Loading easyocr reader for {langs_str}, gpu={gpu_str}, verbose={verbose_str}")
    reader = EasyocrReader(langs,
        model_storage_directory=os.path.join('.cache', 'easyocr'),
        gpu=gpu, verbose=verbose)
    return reader

def good_ch(ch):
    if ch.isalpha():
        return True
    if ch in ''' ,.;:"'!?-''':
        return True
    return False

def is_text_chunk(s, limit=3):
    if len(s) < 2*limit:
        alphas = sum(map(good_ch, s))
        return alphas == len(s)

    counter = 0
    for ch in s:
        if ch.isalpha():
            counter += 1
            if counter >= limit:
                return True
        else:
            counter = 0

    return False

def deserialize_rect(s):
    data = s.split(';')
    data = [ s.split(',') for s in data ]
    xs = [ int(x) for x, _ in data ]
    ys = [ int(y) for _, y in data ]
    data = (min(xs), min(ys), max(xs), max(ys))
    return Rect(data)


class Rect:
    def __init__(self, data=(0,0,0,0)):
        if isinstance(data, str):
            data = deserialize_rect(data)

        if isinstance(data, Rect):
            self.xmin = data.xmin
            self.xmax = data.xmax
            self.ymin = data.ymin
            self.ymax = data.ymax
            return

        if len(data) == 2:
            data = data[0] + data[1]

        if len(data) != 4:
            raise ValueError(f"Invalid data in Rect: {data}")

        x1, y1, x2, y2 = data
        self.xmin = min(x1, x2)
        self.ymin = min(y1, y2)
        self.xmax = max(x1, x2)
        self.ymax = max(y1, y2)

    @property
    def width(self):
        return self.xmax - self.xmin

    @property
    def height(self):
        return self.ymax - self.ymin


class TextChunk(Rect):
    def __init__(self, data):
        super().__init__(data['rect'])

        self.text = data['text']
        self.probability = data['probability']

    def dump(self):
        return f"{self.xmin:4d}, {self.ymin:4d}: {self.text}"

    def dump_rect(self):
        return f"({self.xmin:4d}, {self.ymin:4d}) - ({self.xmax:4d}, {self.ymax:4d}) {self.text}"


class Page:
    def __init__(self, num, png_fn):
        self.num = num
        self.png_fn = png_fn
        self.texts_fn = None
        self.text_chunks = None

    def detect_text_chunks(self, texts_fn, easyocr_reader):
        texts = []
        img = cv2.imread(self.png_fn)
        for coords, text, probability in easyocr_reader.readtext(img):
            coords = [ (int(x), int(y)) for x, y in coords ]
            coords = [ f"{x},{y}" for x, y in coords ]
            texts.append({
                'rect': ';'.join(coords),
                'text': str(text),
                'probability': float(probability),
            })

        with open(texts_fn, 'w', encoding='utf-8') as f:
            dump_yaml(texts, f, allow_unicode=True)

    def load_text_chunks(self, texts_fn):
        chunks = []
        with open(texts_fn, 'r', encoding='utf-8') as f:
            texts = load_yaml(f)
            for data in texts:
                chunks.append(TextChunk(data))

        chunks.sort(key=lambda chunk: (chunk.ymin, chunk.xmin))
        self.texts_fn, self.text_chunks = texts_fn, chunks

    def visualize_text_chunks(self, img_fn):
        img = cv2.imread(self.png_fn)

        def highlight(chunk, r, g, b):
            color = (int(b), int(g), int(r))
            x1, x2, y1, y2 = chunk.xmin, chunk.xmax, chunk.ymin, chunk.ymax
            cv2.line(img, (x1, y1), (x2, y1), color, 5)
            cv2.line(img, (x1, y1), (x1, y2), color, 5)
            cv2.line(img, (x1, y2), (x2, y2), color, 5)
            cv2.line(img, (x2, y1), (x2, y2), color, 5)

        for chunk in self.text_chunks:
            p, q = chunk.probability, 1.0 - chunk.probability
            highlight(chunk, 192*q, 192*p, 0)

        cv2.imwrite(img_fn, img)


class Job:
    def __init__(self, fn, args, *, name=None):
        if name is None:
            basename = os.path.basename(fn)
            name = os.path.splitext(basename)[0]

        self.fn = fn
        self.name = name
        self.dpi = args.dpi
        self.lang_from = args.lang_from
        self.gpu = args.with_gpu
        self.verbose = args.verbose
        self.pages = []

        if self.gpu is None:
            self.gpu = cuda.is_available()
        if isinstance(self.lang_from, list):
            if len(self.lang_from) != 1:
                raise ValueError(f"Invalid lang_from argument: {args.lang_from}")
            self.lang_from = self.lang_from[0]

    def extract_png(self):
        fn, name, dpi = self.fn, self.name, self.dpi
        png_dn = os.path.join('.cache', 'pdfs', name, 'pages', str(dpi))
        os.makedirs(png_dn, exist_ok=True)

        cached = True
        doc = fitz.open(fn)
        qpages = len(doc)
        for num in range(qpages):
            png_fn = os.path.join(png_dn, f"page-{num:03d}.png")
            page = Page(num, png_fn)
            self.pages.append(page)
            if not os.path.exists(png_fn):
                cached = False

        if cached:
            print(f"{name}: use cached PNGs from {png_dn}")
            return

        print(f"{name}: loading PNGs...")
        for page in self.pages:
            doc.load_page(page.num).get_pixmap(dpi=dpi).save(page.png_fn)

    def detect_text_chunks(self):
        name, dpi, lang_from = self.name, self.dpi, self.lang_from
        texts_dn = os.path.join('.cache', 'pdfs', name, 'text-chunks', f"{lang_from}-{dpi}")
        os.makedirs(texts_dn, exist_ok=True)

        todo = []
        for page in self.pages:
            num = page.num
            texts_fn = os.path.join(texts_dn, f"chunks-{num:03d}.yaml")
            if not os.path.exists(texts_fn):
                todo.append((page, texts_fn))

        if not todo:
            print(f"{name}: use cached text chunks from {texts_dn}")
            return

        easyocr_reader = get_easyocr_reader(lang_from, gpu=self.gpu, verbose=self.verbose)
        print(f"{name}: run easyocr to identity text chunks...")
        for page, texts_fn in todo:
            page.detect_text_chunks(texts_fn, easyocr_reader)

    def load_text_chunks(self, *, debug=False):
        name, dpi, lang_from = self.name, self.dpi, self.lang_from
        texts_dn = os.path.join('.cache', 'pdfs', name, 'text-chunks', f"{lang_from}-{dpi}")
        if debug:
            debug_dn = os.path.join('.cache', 'pdfs', name, 'debug', 'text-chunks', f"{lang_from}-{dpi}")
            os.makedirs(debug_dn, exist_ok=True)

        print(f"{name}: load cached text chunks from {texts_dn}")
        for page in self.pages:
            num = page.num
            texts_fn = os.path.join(texts_dn, f"chunks-{num:03d}.yaml")
            page.load_text_chunks(texts_fn)
            if debug:
                img_fn = os.path.join(debug_dn, f"chunks-{num:03d}.png")
                page.visualize_text_chunks(img_fn)

    def process(self):
        self.extract_png()
        self.detect_text_chunks()
        self.load_text_chunks()

def main():
    parser = ArgumentParser(description='Mustitz PDF autotranslation minitool.')

    def_arg = parser.add_argument
    def_arg('files', nargs='+', help='Language and PDF file names')
    def_arg('--dpi', nargs=1, type=int, default=600, help='DPI for saved PDF pages')
    def_arg('--lang-from', nargs=1, type=str, default='en', help='Input PDF Language')
    def_arg('--verbose', default=False, action='store_true', help='Verbose logging')
    def_arg('--with-gpu', default=None, action='store_true', help='GPU support ON')
    def_arg('--without-gpu', dest='with_gpu', action='store_false', help='GPU support OFF')

    args = parser.parse_args()
    for fn in args.files:
        job = Job(fn, args)
        job.process()


if __name__ == '__main__':
    main()
