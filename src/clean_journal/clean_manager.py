from __future__ import division, print_function, absolute_import
import multiprocessing as mp
from functools import partial
import subprocess
import os
import cPickle as pickle
import copy_reg
import types
import logging
import argparse
import time
from src.utilities import count_lines, split_file, _pickle_method, _unpickle_method
from src.clean_journal.clean_worker import JournalCleaningWorker
from src.journal import Journal

logger = logging.getLogger(__name__)
# need to change how pickling is done to run class objects in parallel
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class JournalCleaningManager(object):
    """
    Provides functionality to create 'journals' objects that
    partition all the data and clean them using parallel nodes,
    and then combines the results into a single flat file.

    TODO:
    1. may want final result create by process_journals() to be a bag of words file and a vocabulary,
       instead of a flat file with the cleaned text?

    2. it would be cleaner to have clean_method be an argument passed to
       the clean_journal() method of Journals, rather than a class variable
       of Journals. This would require passing function arguments in the
       multiprocessing code -- how is this done?
    """
    def __init__(self, input_dir, output_file, clean_method='topic', n_workers=1, verbose=True):
        """
        Args:
            input_dir: Folder containing all of the sites folders. For example:
                       /home/srivbane/shared/caringbridge/data/parsed_json
            output_file: Name of final result containing all cleaned data

            clean_method: Can be any one of ('topic', 'sentiment', 'survival', 'none'), specifying
                       the method of cleaning the text (either for topic modeling, sentiment
                       analysis, survival analysis, or no cleaning, respectively
            n_workers: The number of nodes to use to process the data. Defaults to 1 node.
            verbose:   Should progress be printed to the log? True/False
        Returns: Nothing, just initializes the class instance.
        """
        self.input_dir = input_dir
        self.output_file = output_file
        self.clean_method = clean_method
        self.n_workers = n_workers
        self.verbose = verbose

        # save intermediate files in the same directory as final output
        self.output_dir = os.path.split(output_file)[0]

        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def clean_journals(self):
        """
        Main function for processing the json files in parallel.
        This recruits n_workers to work on the journal data
        """
        # get the list of shard files in the input_dir
        input_files = [os.path.join(self.input_dir, fname) for fname in os.listdir(self.input_dir)]

        # clean an write all the journal shards to files
        cleaned_shards = self.assign_tasks(input_files)

        # concatenate the resulting shards into a single big file
        self.concatenate_shards(cleaned_shards)

        # remove the cleaned shards
        #self.remove_shards(cleaned_shards)

        # sort the final result
        self.sort_result()

    def assign_tasks(self, input_files):
        """
        This calls the workers in parallel (if requested)
        """
        # what should the resulting files be named?
        fnames = [fname for fpath, fname in [os.path.split(i) for i in input_files]]
        output_files = [os.path.join(self.output_dir, i.replace("parsed_", "cleaned_")) for i in fnames]

        if self.n_workers == 1 or len(input_files) == 1:
            logger.info("No parallel processing needed, only one worker requested.")
            for input_file, output_file in zip(input_files, output_files):
                worker = JournalCleaningWorker(input_file=input_file, output_file=output_file, clean_method=self.clean_method, init_stream=True, verbose=self.verbose)
                worker.clean_and_save()
        else:
            logger.info("Multiple files given, processing them with " + str(self.n_workers) + " workers.")
            logger.info("Number of input files: " + str(len(input_files)))
            logger.info("Number of output fies: " + str(len(output_files)))

            # create instructions to pass to each worker
            args_list = zip(input_files, output_files)
            # initialize worker instances
            processes = [JournalCleaningWorker(input_file=i, output_file=o, clean_method=self.clean_method, init_stream=False, verbose=self.verbose) for i, o in args_list]
            # intialize multiprocessing pool and tell works to run the clean_and_save method
            pool = mp.Pool(processes=self.n_workers)
            function_call = partial(JournalCleaningWorker.clean_and_save)
            pool.map(function_call, processes)
            pool.close()
        return output_files

    def concatenate_shards(self, filenames):
        """
        combine a list of files into one big file
        use command line tools
        """
        cmd = "cat "
        for f in filenames:
            cmd += f + " "

        cmd += "> " + self.output_file
        try:
            subprocess.check_call(cmd, shell=True)
        except subprocess.CalledProcessError as e:
            raise Exception("Couldn't concatenate these files: " + ', '.join(filenames))

    def remove_shards(self, filenames):
        for f in filenames:
            os.remove(f)

    def sort_result(self, pct_memory="50%"):
        """
        Sort based on siteId (first column) and createdAt (4th column)
        """
        cmd = """/bin/bash -c "sort %s -t $'\t' -k1,1 -k4,4 -o %s -S %s" """ % (self.output_file, self.output_file, pct_memory)
        subprocess.call(cmd, shell=True)




def main():
    parser = argparse.ArgumentParser(description='This program recruits JournalCleaningWorkers to do the journal cleaning in parallel.')
    parser.add_argument('-i', '--input_dir', type=str, help='Path to directory containing all the site directories.')
    parser.add_argument('-n', '--n_workers', type=int, default=1, help='Number of workers / CPUs to use to process the journal files.')
    parser.add_argument('-c', '--clean_method', type=str, default='topic', help='Method of cleaning each journal. Default is the topic modeling cleaning method "topic".')
    parser.add_argument('-o', '--output_file', type=str, help='Name of file to write out all the results to.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    print('clean_manager.py')
    print(args)

    start = time.time()
    jm = JournalCleaningManager(input_dir=args.input_dir, output_file=args.output_file, clean_method=args.clean_method, n_workers=args.n_workers, verbose=args.verbose)
    jm.clean_journals()
    end = time.time()
    print("Time to process the files:", end - start, "seconds")

    print("Printing top rows of output:", "\n")
    with open(args.output_file, 'r') as fin:
        for i, line in enumerate(fin):
            if i > 5:
                break

            print(line.replace('\n', ''))




if __name__ == "__main__":
    main()
