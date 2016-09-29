from __future__ import division, print_function
import os
import logging
import os
import argparse
import multiprocessing as mp
from math import ceil
from functools import partial
import cPickle as pickle
import copy_reg
import types
from parse_worker import JournalParsingWorker
from utilities import split_json_file
import time

logger = logging.getLogger(__name__)


# need to change how pickling is done to run class objects in parallel
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)


class JournalParsingManager(object):
    """
    JournalParsingManager is the primary method for parsing
    the journal file.
    """
    def __init__(self, n_workers, verbose):
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
        self.verbose = verbose
        self.n_workers = n_workers
            
    def check_file_splits(self, input_file, input_file_length=None):
        # check if the file has already been split
        input_file_label = input_file.split('/')[-1].replace('.json', '')
        shard_dir = os.path.join('/'.join(input_file.split('/')[0:-1]), input_file_label + '_shards')
        file_exists = []
        shard_filenames = []
        for i in range(self.n_workers):
            shard_file = os.path.join(shard_dir, input_file_label + '_' + str(i+1).zfill(2) + '_of_' + str(self.n_workers) + '.json')
            shard_filenames.append(shard_file)
            file_exists.append(os.path.isfile(shard_file))

        if all(file_exists):
            logger.info('Looks like the file splits already exist, no need to re-split')
        else:
            logger.info('Splitting the file now')
            split_json_file(input_file, self.n_workers, input_file_length)

        return shard_filenames
    
    def parse_files(self, input_file, output_dir, input_file_length=None):
        """
        Main function for processing the json files in parallel.
        This recruits n_workers to work on the json data
        """
        if self.n_workers == 1:
            # no need to split file
            input_files = [input_file]
        else:
            # split the file so there is work availabe for each worker
            start = time.time()
            input_files = self.check_file_splits(input_file, input_file_length)
            end = time.time()
            logger.info("Time to split the files into shards " + str(end - start))
        
        if len(input_files) == 1:
            logger.info("No parallel processing needed, only one file for one worker")
            worker = JournalParsingWorker(input_path=input_files[0], output_dir=output_dir, verbose=self.verbose)
            num_skipped = worker.parse_file()
        else:
            logger.info("Multiple files given, processing them with " + str(self.n_workers) + " workers.")
            # create instructions to pass to each worker
            n_shards = len(input_files)
            args_list = zip(input_files, [output_dir]*n_shards, [self.verbose]*n_shards)
            processes = [JournalParsingWorker(input_path=i, output_dir=o, verbose=v) for i, o, v in args_list]
            pool = mp.Pool(processes=self.n_workers)
            function_call = partial(JournalParsingWorker.parse_file)
            num_skipped_list = pool.map(function_call, processes)
            pool.close()
            num_skipped = sum(num_skipped_list)
        logger.info("Total number of (deleted + draft) journal entries skipped: " + str(num_skipped))


def main():
    import subprocess
    parser = argparse.ArgumentParser(description='Main program for calling multiple workers to parse the journals.json file.')
    parser.add_argument('-i', '--input_file', type=str, help='Name of the journal file you want parsed..')
    parser.add_argument('--num_lines', type=int, help='Number of lines in the input_file. Specifying this can speed up performance of splitting the file into shards.')
    parser.add_argument('-o', '--output_dir', type=str, help='Name of output directory for where to create site directories.')
    parser.add_argument('-n', '--n_workers', type=int, help='How many processors to work on the journal file')
    parser.add_argument('--log', dest="verbose", action="store_true", help="Add this flag to have progress printed to the log.")
    parser.add_argument('--clean', dest='clean', action='store_true', help='Add this flag to remove all site directories before running the program')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print('parse_manager.py')
    print(args)

    if args.clean:
        print('Removing data from a previous run...')
        cmd = 'rm -rf /home/srivbane/shared/caringbridge/data/parsed_json'
        subprocess.call(cmd, shell=True)
        subprocess.call('mkdir /home/srivbane/shared/caringbridge/data/parsed_json', shell=True)
        print("Remove shard files from previous run...")
        cmd = "rm -rf " + args.input_file.replace('.json', '_shards')
        subprocess.call(cmd, shell=True)

    start = time.time()
    manager = JournalParsingManager(n_workers=args.n_workers, verbose=args.verbose)
    manager.parse_files(input_file=args.input_file, output_dir=args.output_dir, input_file_length=args.num_lines)
    end = time.time()
    print("Time to parse the file:", end - start, "seconds.")
    
if __name__ == "__main__":
    main()
