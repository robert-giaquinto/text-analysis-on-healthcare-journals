from __future__ import division, print_function, absolute_import
import argparse

from src.clean_journal.clean_worker import JournalCleaningWorker


def main():
    parser = argparse.ArgumentParser(description='Each worker contains the methods used to clean the journal data for analysis. Workers can be called to work in parallel via clean_manager.py.')
    parser.add_argument('-i', '--input_file', type=str, help='File with a journal on each line.')
    parser.add_argument('-o', '--output_file', type=str, help='Where to save the cleaned results.')
    parser.add_argument('-c', '--clean_method', type=str, default='topic', help='Method of cleaning each journal. Default is the topic modeling cleaning method "topic".')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    print('clean_worker.py')
    print(args)

    j = JournalCleaningWorker(input_file=args.input_file, output_file=args.output_file, clean_method=args.clean_method, init_stream=True, verbose=args.verbose)
    
    for i, journal in enumerate(j.stream):
        if i > 3:
            break
        print("Original journal:\n")
        print(journal)
        print("\nCleaned journal:\n")
        print(j.clean_journal(journal))


if __name__ == "__main__":
    main()
