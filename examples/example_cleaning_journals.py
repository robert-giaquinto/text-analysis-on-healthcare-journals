from __future__ import division, print_function
import argparse
from journals import Journals, Journal


def main():
    # where are the data location?
    sites_dir = "/home/srivbane/shared/caringbridge/data/parsed_json/"
    keys_file = "/home/srivbane/shared/caringbridge/data/clean_journals/all_keys.tsv"

    # create the object that iterates over journal entries
    j = Journals(sites_dir=sites_dir, keys_file=keys_file, verbose=True)

    print("Iterating over the stream of journal entries")
    print("and applying the journal cleaning method defined")
    print("in Journals.clean_journal()")
    for i, journal in enumerate(j.stream):
        # only do this for a few journal
        if i > 2:
            break

        # perform the data cleaning
        body = journal.body # save original for comparison
        cleaned_journal = j.clean_journal(journal)
        print(cleaned_journal)
        print("Body before cleaning:", body, "\n")
    


if __name__ == "__main__":
    main()
