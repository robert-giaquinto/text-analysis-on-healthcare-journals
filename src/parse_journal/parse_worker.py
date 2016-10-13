from __future__ import division, print_function, absolute_import
import re
import os
import logging
import json
import argparse
logger = logging.getLogger(__name__)

# TODO: It may be good to delete all site folders before running this program again

class JournalParsingWorker(object):
    """
    Works on a single "shard" of the journal.json file to
    parse the json into a tab delimited format
    """
    def __init__(self, input_path, output_dir,  verbose):
        """
        Iterate through the file input_path, for each journal entry:
        1) create a directory for the site, if it doesn't already exist
        2) create a file in that directory for the journal entry

        NOTE: Going back to this approach because the journal.json file
              isn't sorted in any meaningful way, so pulling out all the
              journal entries for one site at a time isn't possible.

              Parse the data in the way this file works at least gets
              all the journals for each site in one location.

              We can combine journal entries into a single file at the next
              step.

        Args:
            input_path: full filename of input file.
                ex: /home/srivbane/shared/caringbridge/data/dev/journal.json
            output_dir: directory where the site-directories should be created
                ex: /home/srivbane/shared/caringbridge/data/parsed_json/
            verbose: True/False where to print progress to the log file.
        Returns: Nothing, it simply writes each journal entry to a file in an
            appropriate directory.
        """
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.input_path = input_path
        self.output_dir = output_dir

        # which fields from the journal.json file should we keep (store them in output file names)
        self.fields = ['siteId', 'userId', 'journalId', 'createdAt']
        self.no_journalId_count = 0
        self.no_userId_count = 0

    def parse_file(self):
        """
        Primary function to call, this does all the work
        """
        logger.info('Opening file: ' + self.input_path)
        num_skipped = 0
        with open(self.input_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i % 100000 == 0:
                    logger.info('Processing journal: ' + str(i))

                # parse the json
                json_dict = json.loads(line)

                # put in checks to see if journal should be skipped (i.e. deleted=True, draft=True)
                skip = self.check_skip(json_dict)
                if skip:
                    num_skipped += 1
                    continue

                # check if journalId doesn't exist, if not make up a unique id
                if 'journalId' not in json_dict:
                    json_dict['journalId'] = '-1'
                    self.no_journalId_count += 1

                # check if userId doesn't exist, if so make one up
                if 'userId' not in json_dict:
                    json_dict['userId'] = '-1'
                    self.no_userId_count += 1

                self.check_directory(json_dict['siteId'])

                # open a new file for this journal entry and paste in the text
                self.save_journal(json_dict)

        logger.info("Had to make-up a userId for " + str(self.no_userId_count) + "journals.")
        logger.info("Had to make-up a journalId for " + str(self.no_journalId_count) + "journals.")
        return num_skipped

    def check_skip(self, json_dict):
        """
        Check to see if this journal should be skipped.
        """
        any_deletes = 0
        if 'isDeleted' in json_dict:
            any_deletes += json_dict['isDeleted'] == "1"

        if 'isDraft' in json_dict:
            any_deletes += json_dict['isDraft'] == "1"

        # is there a more efficient way to perform this check?
        if 'body' in json_dict:
            any_deletes += json_dict['body'].strip() == "This CaringBridge site was created just recently. Please visit again soon for a journal update."
        else:
            any_deletes += 1 # remove journals with no text

        # remove any journals without a timestamp
        if 'createdAt' not in json_dict:
            any_deletes += 1

        # remove any journals without a timestamp
        if 'siteId' not in json_dict:
            any_deletes += 1

        return any_deletes > 0

    def check_directory(self, site_id):
        site_dir = os.path.join(self.output_dir, str(site_id))
        if not os.path.isdir(site_dir):
            try:
                os.makedirs(site_dir)
            except OSError:
                pass

    def save_journal(self, json_dict):
        """
        Save the desired fields of the json data to a tab delimited file
        """
        filename_list = []
        for field in self.fields:
            if field == 'createdAt':
                filename_list.append(str(json_dict[field]['$date']))
            else:
                filename_list.append(str(json_dict[field]))
        output_filename = '_'.join(filename_list)
        journal_file = os.path.join(self.output_dir, str(json_dict['siteId']), output_filename)

        # write out the text int the journal entry, include title (if exists) at begining.
        text = json_dict['body'].encode('utf-8').strip()
        if 'title' in json_dict:
            text = json_dict['title'].encode('utf-8').strip() + ' ' + text

        with open(journal_file, 'wb') as f:
            f.write(text)




def main():
    import subprocess
    parser = argparse.ArgumentParser(description='Example for how to run the parse worker to extract journals from a journal.json file.')
    parser.add_argument('-i', '--input_path', type=str, help='Name of journal file to parse.')
    parser.add_argument('-o', '--output_dir', type=str, help='Name of output directory to create site directories.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print('parse_worker.py')
    print(args)

    worker = JournalParsingWorker(input_path=args.input_path, output_dir=args.output_dir, verbose=args.verbose)
    worker.parse_file()

if __name__ == "__main__":
    main()
