from __future__ import division, print_function
import multiprocessing as mp
from functools import partial
import os
import cPickle as pickle
import copy_reg
import types
import logging
import argparse
from parse_journal.utilities import count_lines, split_file, _pickle_method, _unpickle_method
from journals import Journal, Journals

logger = logging.getLogger(__name__)
# need to change how pickling is done to run class objects in parallel
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class JournalsManager(object):
    """
    Provides functionality to create journals
    objects that clean multiple shards of the text
    data in parallel, and then combines the results
    """
    def __init__(self, sites_dir, keys_file, n_workers=1, verbose=True):
        """
        Args:
            sites_dir: Folder containing all of the sites folders. For example:
                       /home/srivbane/shared/caringbridge/data/parsed_json
            keys_file: the name of the tsv file containing the keys of all the journal
                       entries. If this file doesn't exists, it will be created.
                       Example:
                       /home/srivbane/shared/caringbridge/data/cleaned_journals/all_keys.tsv
            n_workers: The number of nodes to use to process the data. Defaults to 1 node.
            verbose:   Should progress be printed to the log? True/False
        Returns: Nothing, just initializes the class instance.
        """
        self.sites_dir = sites_dir
        self.n_workers = n_workers
        self.verbose = verbose
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        
        # make sure the keys_file is usable:
        self.init_keys_file(keys_file)
        # partition the keys file so there will be journals assigned to each worker:
        self.keys_list = self.partition_keys()
        
    def init_keys_file(self, keys_file):
        """
        check is a keys file is given (and if it really exists), if not create one

        Args:
            keys_file: full filename of the tsv file containing keys to each joural
                       if this file doesn't exists, create it
                       if no filename is given, create the file (TODO: this is possbibly a bad default...)
        Returns: Nothing, just makes sure that self.keys_file is a legitmate file we can use
        """
        if keys_file is None:
            logger.info("No key's file given. Creating keys for all directories in sites_dir")
            self.keys_file = "/home/srivbane/shared/caringbridge/data/clean_journals/all_keys.tsv"
            kc = KeyCollector(input_dir=self.sites_dir, output_filename=self.keys_file, verbose=self.verbose)
            kc.collect_keys()
        elif not os.path.isfile(keys_file):
            logger.info("The key's file given doesn't exists. Creating keys for all directories in sites_dir")
            self.keys_file = keys_file
            kc = KeyCollector(input_dir=self.sites_dir, output_filename=self.keys_file, verbose=self.verbose)
            kc.collect_keys()
        else:
            self.keys_file = keys_file
            logger.info("Keys file found, using the file you specified.")
            
    def partition_keys(self, chunks_per_worker=10):
        """
        Partition the flat file of journal keys into enough parts
        so that each worker will process a few parts of the file.

        More partitions = smaller memory constraints. If you want to
        only hold 10% of the data in memory at a time set chunks_per_worker
        to 10. For 20% of data memory at any time use chunks_per_worker=5. 
        """
        # perform some check that the keys_file is legitmate

        # determine the division of the keys_file
        n_partitions = self.n_workers * chunks_per_worker
        if n_partitions == 1:
            return [self.keys_file]
        
        # write each partition of the keys to a new file
        keys_dir = '/'.join(self.keys_file.split('/')[0:-1])
        keys_prefix, keys_ext = os.path.splitext(self.keys_file.split('/')[-1])
        shard_filenames = split_file(self.keys_file, n_splits=n_partitions)
        return shard_filenames

    def process_journals(self, outfile):
        """
        Main function for processing the json files in parallel.
        This recruits n_workers to work on the journal data

        TODO:
        instead of writing all the text to a file
        this could create the dictionary object (Gensim) and
        the bag of words file. Go with this approach if we
        decide we don't need the intermediate data of cleaned texts
        """
        if self.n_workers == 1:
            logger.info("No parallel processing needed, only one worker requested.")
            with open(outfile, 'wb') as fout:
                for keys_file in self.keys_list:
                    j = Journals(sites_dir=self.sites_dir, keys_file=keys_file, init_stream=True, verbose=self.verbose)
                    for i, journal in enumerate(j.stream):
                        journal = j.clean_journal(journal)
                        fout.write(' '.join(journal.body) + '\n')
            
        else:
            logger.info("Multiple files given, processing them with " + str(self.n_workers) + " workers.")
            # create instructions to pass to each worker
            processes = [Journals(sites_dir=self.sites_dir, keys_file=k, init_stream=False, verbose=self.verbose) for k in self.keys_list]
            pool = mp.Pool(processes=self.n_workers)

            # tell each worker to run the process_journals function
            # which will clean all the journals assigned to the worker
            # and return the results
            function_call = partial(Journals.process_journals)

            # combine results from all workers into a single (big) file
            with open(outfile, 'wb') as fout:
                for journal_collection in pool.imap(function_call, processes):
                    for journal in journal_collection:
                        fout.write(journal.siteId + '\t' + journal.userId + '\t' + journal.journalId + '\t' + journal.createdAt + '\t' + ' '.join(journal.body) + '\n')
            
            pool.close()


                    
def main():
    parser = argparse.ArgumentParser(description='This program runs the Journals cleaning method in parallel and combines the results.')
    parser.add_argument('-i', '--sites_dir', type=str, help='Path to directory containing all the site directories.')
    parser.add_argument('-k', '--keys_file', type=str, help='Filename containing the keys to the journals files. Default=None results in using all journals in sites_dir.')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers / CPUs to use to process the journal files.')
    parser.add_argument('-o', '--outfile', type=str, help='Name of file to write out all the results to.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    
    print('journals_manager.py')
    print(args)

    jm = JournalsManager(sites_dir=args.sites_dir, keys_file=args.keys_file, n_workers=args.n_workers, verbose=args.verbose)
    jm.process_journals(outfile=args.outfile)

    print("Printing top rows of output:", "\n")
    with open(args.outfile, 'r') as fin:
        for i, line in enumerate(fin):
            if i > 25:
                break

            print(line.replace('\n', ''))




if __name__ == "__main__":
    main()
