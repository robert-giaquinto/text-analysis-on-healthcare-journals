from __future__ import division, print_function
import argparse
from journals import Journals, Journal


def journal_cleaner(journal_text):
    """
    example of a function to clean the journal text
    """
    rval = journal_text.replace('\n', '')
    return rval


def main():
    # where are the data location?
    sites_dir = "/home/srivbane/shared/caringbridge/data/parsed_json/"
    keys_file = "/home/srivbane/shared/caringbridge/data/parsed_json/all_keys.tsv"

    # create the object that iterates over journal entries
    j = Journals(sites_dir=sites_dir, keys_file=keys_file)
    gen = j.journal_generator()

    for i, journal in enumerate(gen):
        # only do this for a few journal
        if i > 3:
            break

        # perform the data cleaning
        cleaned_body = journal_cleaner(journal.body)
        print(cleaned_body, "\n")
    


if __name__ == "__main__":
    main()
