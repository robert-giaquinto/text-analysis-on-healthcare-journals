from __future__ import division, print_function, absolute_import
import os
import argparse

from src.utilities import split_file

def main():
    parser = argparse.ArgumentParser(description='Split file.')
    parser.add_argument('--filename', type=str)
    parser.add_argument('--n_splits', type=int)
    parser.add_argument('--infile_len', type=int)
    args = parser.parse_args()

    rval = split_file(filename=args.filename, n_splits=args.n_splits, infile_len=args.infile_len)
    print(rval)


if __name__ == "__main__":
    main()
