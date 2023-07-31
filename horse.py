from argparse import ArgumentParser

def process_file(fn):
    print(f"Not implemented process_file: {fn}")

def main():
    parser = ArgumentParser(description='Mustitz PDF autotranslation minitool.')

    def_arg = parser.add_argument
    def_arg('files', nargs='+', help='PDF file names')

    args = parser.parse_args()
    for fn in args.files:
        process_file(fn)

if __name__ == '__main__':
    main()
