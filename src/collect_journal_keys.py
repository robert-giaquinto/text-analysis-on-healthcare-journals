from __future__ import division, print_function, absolute_import
import os
import argparse
import logging
logger = logging.getLogger(__name__)


class KeyCollector(object):
    """
    collect the keys to all the journal files and store in a single flat file
    """
    def __init__(self, input_dir, output_filename, verbose=False):
        self.input_dir = input_dir
        self.output_filename = output_filename

        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    
    def collect_keys(self, sort_results=True):
        """
        main function for performing key collection process
        """
        # get a list of all the site directories
        site_dirs = [d for d in os.listdir(self.input_dir) if os.path.isdir(os.path.join(self.input_dir, d))]
        if sort_results:
            site_dirs.sort(key=lambda x: int(x))
        
        with open(self.output_filename, 'wb') as fout:
            for siteId in site_dirs:
                logger.info("Collecting keys for site "+ siteId)

                # find all journal entries for this site
                site_path = os.path.join(self.input_dir, siteId)
                journal_filenames = os.listdir(site_path)

                # loop through each filename and save the journal's keys to the flat file
                for journal_filename in journal_filenames:
                    # parse the filename to extract key information
                    journal_keys = journal_filename.split('_')
                    fout.write('\t'.join(journal_keys) + '\n')


def main():
    parser = argparse.ArgumentParser(description='Example of how to collect keys on all journals that have been parsed, and put into a flat file.')
    parser.add_argument('-i', '--input_dir', type=str, help='Path to directory containing all the site directories.')
    parser.add_argument('-o', '--output_filename', type=str, help='Full filename to for where to save all the keys found..')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')

    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print('collect_journal_keys.py')
    print(args)

    kc = KeyCollector(input_dir=args.input_dir, output_filename=args.output_filename, verbose=args.verbose)
    kc.collect_keys()


if __name__ == "__main__":
    main()
