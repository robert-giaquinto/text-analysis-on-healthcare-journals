from __future__ import division, print_function
import os
import logging
import argparse
from parse_journal.collect_journal_keys import KeyCollector

logger = logging.getLogger(__name__)


# TODO should probabily check that the journal files don't contain carriage returns

class Journals(object):
    """
    provides an iterator over journal entries
    """
    def __init__(self, sites_dir, keys_file=None, verbose=False):
        """
        Args:
            sites_dir: Folder where all of the folders for each site exist
            keys_file: File containing a list of the keys for all the journal files
                       you want to iterate over.
                       If no argument given, a keys file will be created for all sites
                       in the sites_dir.
            verbose:   Flag for whether or not to print progress to the log.
        """
        
        self.sites_dir = sites_dir
        self.verbose = verbose
        
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        # check is a keys file is given (and if it really exists), if not create one
        if keys_file is None:
            logger.info("No key's file given. Creating keys for all directories in sites_dir")
            self.keys_file = os.path.join(sites_dir, "all_keys.tsv")
            kc = KeyCollector(input_dir=sites_dir, output_filename=self.keys_file, verbose=verbose)
            kc.collect_keys()
        elif not os.path.isfile(keys_file):
            logger.info("The key's file given doesn't exists. Creating keys for all directories in sites_dir")
            self.keys_file = keys_file
            kc = KeyCollector(input_dir=sites_dir, output_filename=self.keys_file, verbose=verbose)
            kc.collect_keys()
        else:
            self.keys_file = keys_file
            logger.info("Keys file found, using the file you specified.")


    def journal_generator(self):
        """
        Return a generator over the journal files
        
        ex:
        journals = Journals(sites_dir = '/home/srivbane/shared/caringbridge/data/parsed_json')
        jg = journals.journal_generator()
        for journal in jg:
            print(journal.siteId)
            print(journal.journalId)
            print(journal.body)
        """
        with open(self.keys_file, 'r') as key_file:
            for line in key_file:
                # pull out the keys to a journal and unpack the values into variables 
                journal_keys = line.replace('\n', '').strip().split('\t')
                siteId, userId, journalId, createdAt = journal_keys
                
                filename = os.path.join(self.sites_dir, siteId, '_'.join(journal_keys))
                with open(filename, 'r') as journal_file:
                    # body of journal may be spread out of multiple lines, remove all blank lines
                    body = filter(None, [j.replace('\n', ' ').strip() for j in journal_file.readlines()])
                    # paste everything back together
                    body = '\n'.join(body)
                    
                journal = Journal(siteId=siteId, userId=userId, journalId=journalId, createdAt=createdAt, body=body)
                yield journal


class Journal(object):
    """
    Just a simple structure to hold journal information
    """
    def __init__(self, siteId=None, userId=None, journalId=None, createdAt=None, body=None):
        self.siteId = siteId
        self.userId = userId
        self.journalId = journalId
        self.createdAt = createdAt
        self.body = body

    def __repr__(self):
        return "Journal Object"

    def __str__(self):
        return "\nsiteId: " + str(self.siteId) +\
            "\nuserId: " + str(self.userId) +\
            "\njournalId: " + str(self.journalId) +\
            "\ncreatedAt: " + str(self.createdAt) +\
            "\nbody: " + self.body
        
def main():
    parser = argparse.ArgumentParser(description='An example of using the iterator over journal files.')
    parser.add_argument('-i', '--sites_dir', type=str, help='Path to directory containing all the site directories.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    
    print('journals.py')
    print(args)

    j = Journals(sites_dir=args.sites_dir)
    gen = j.journal_generator()

    for i, journal in enumerate(gen):
        if i > 3:
            break
        print(journal)




if __name__ == "__main__":
    main()
