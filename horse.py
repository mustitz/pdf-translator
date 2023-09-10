import os

from argparse import ArgumentParser
from collections import namedtuple

import fitz


Page = namedtuple('Page', ('num', 'png_fn'))


class Job:
    def __init__(self, fn, args, *, name=None):
        if name is None:
            basename = os.path.basename(fn)
            name = os.path.splitext(basename)[0]

        self.fn = fn
        self.name = name
        self.dpi = args.dpi
        self.pages = []

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

    def process(self):
        self.extract_png()


def main():
    parser = ArgumentParser(description='Mustitz PDF autotranslation minitool.')

    def_arg = parser.add_argument
    def_arg('files', nargs='+', help='PDF file names')
    def_arg('--dpi', nargs=1, type=int, default=600, help='DPI for saved PDF pages')

    args = parser.parse_args()
    for fn in args.files:
        job = Job(fn, args)
        job.process()


if __name__ == '__main__':
    main()
