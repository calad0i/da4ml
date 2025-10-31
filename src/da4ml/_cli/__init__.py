import argparse

from .. import _version
from .convert import _add_convert_args, convert_main
from .report import _add_report_args, report_main


def main():
    parser = argparse.ArgumentParser(description='Welcome to the da4ml command line interface')
    subparsers = parser.add_subparsers(dest='command')

    convert_parser = subparsers.add_parser('convert', help='Convert a Keras model to RTL project')
    report_parser = subparsers.add_parser('report', help='Generate report from an existing RTL projects')
    _add_convert_args(convert_parser)
    _add_report_args(report_parser)
    parser.add_argument('--version', '-v', action='version', version=f'%(prog)s {_version.__version__}')
    args = parser.parse_args()

    match args.command:
        case 'convert':
            convert_main(args)
        case 'report':
            report_main(args)
        case _:
            parser.print_help()
            exit(1)


if __name__ == '__main__':
    main()
