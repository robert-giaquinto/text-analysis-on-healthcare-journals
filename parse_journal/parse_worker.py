from __future__ import division, print_function
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

    def parse_file(self):
        """
        Primary function to call, this does all the work
        """
        logger.info('Opening file: ' + self.input_path)
        with open(self.input_path, 'r') as fin:
            for i, line in enumerate(fin):
                if i % 100 == 0:
                    logger.info('Processing journal: ' + str(i))
                
                # parse the json
                json_dict = json.loads(line)

                # put in checks to see if journal should be skipped (i.e. deleted=True, draft=True)
                skip = self.check_skip(json_dict)
                if skip:
                    logger.info('Skipping:\n' + line)
                    continue

                # check if a site exists for the site_id, create directory if not
                if 'siteId' not in json_dict:
                    # might be safe to just continue here
                    raise ValueError("No siteId found for:\n" + line)
                self.check_directory(json_dict['siteId'])

                # open a new file for this journal entry and paste in the text
                self.save_journal(json_dict)

    def check_skip(self, json_dict):
        """
        Check to see if this journal should be skipped.
        """
        if 'isDeleted' in json_dict:
            return json_dict['isDeleted'] == "1"
        if 'isDraft' in json_dict:
            return json_dict['isDraft'] == "1"
        return False

    def check_directory(self, site_id):
        site_dir = os.path.join(self.output_dir, str(site_id))
        if not os.path.isdir(site_dir):
            os.makedirs(site_dir)
    
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
        journal_file = os.path.join(self.output_dir, str(json_dict['siteId']), 'journal_' + output_filename)
        title_file = os.path.join(self.output_dir, str(json_dict['siteId']), 'title_' + output_filename)

        # write out the journal entry to a file
        if 'body' in json_dict:
            with open(journal_file, 'wb') as j:
                j.write(json_dict['body'].encode('utf-8').strip())

        # write out the title of the journal entry to a file
        if 'title' in json_dict:
            with open(title_file, 'wb') as t:
                t.write(json_dict['title'].encode('utf-8').strip())
                  
        


def main():
    import subprocess
    parser = argparse.ArgumentParser(description='Main progam for running topic modeling experiments.')
    parser.add_argument('-i', '--input_path', type=str, help='Name of journal file to parse.')
    parser.add_argument('-o', '--output_dir', type=str, help='Name of output directory to create site directories.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--clean', dest='clean', action='store_true', help='Add this flag to remove all site directories before running the program')
    parser.set_defaults(clean=False)
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print('parse_worker.py')
    print(args)

    if args.clean:
        print('Removing data from a previous run...')
        cmd = 'rm -rf /home/srivbane/shared/caringbridge/data/parsed_json'
        subprocess.call(cmd, shell=True)
        subprocess.call('mkdir /home/srivbane/shared/caringbridge/data/parsed_json', shell=True)

    worker = JournalParsingWorker(input_path=args.input_path, output_dir=args.output_dir, verbose=args.verbose)
    worker.parse_file()

if __name__ == "__main__":
    main()
