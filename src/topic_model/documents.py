from __future__ import division, print_function, absolute_import
import os
from gensim import corpora
import time
import argparse
import logging

from journal_tokens import JournalTokens

logger = logging.getLogger(__name__)


class Documents(object):
    """
    Iterate over the cleaned journal tokens to create a bag-of-words
    dataset, and a vocabulary
    
    Implementing this as a class to keep track of all the variables
    used creating the vocabulary, and storing/loading the BOW data
    without needed to rebuild the BOW every time
    """
    def __init__(self, journal_file, data_dir=None, keep_n=25000, rebuild=True, verbose=False):
        """
        Args:
            journal_file: full file name of the big flat file of all tokenized journals and their keys.
            data_dir: folder to save the output vocab and bow files.
            keep_n: number of terms to keep in the vocabulary.
            rebuild: true to rewrite the output files, false to use existing files (if they exist)
            verbose: print progress to log
        Return: None
        """
        if data_dir is None:
            self.data_dir = os.path.dirname(journal_file).replace("clean_journals", "topic_model")
        else:
            self.data_dir = data_dir
        
        # init variable to store the vocab in memory
        self.vocab = None
        self.vocab_file = os.path.join(self.data_dir, "vocab_" + str(keep_n) + ".dict")
        # init generator over the bag of words file
        self.bow = None
        # name of file where MatrixMarket bag-of-words data is stored
        self.mm_file = os.path.join(self.data_dir, "bow_" + str(keep_n) + ".mm")
        # save how many documents are stored in the MM file
        self.num_docs = None

        self.journal_file = journal_file
        self.keep_n = keep_n
        self.rebuild = rebuild

        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    def fit(self):
        """
        iterate over the journal file and create the BOW of file
        and the vocabulary
        """
        if self.rebuild or not os.path.isfile(self.vocab_file):
            # build the vocabulary file
            logger.info("Building a new vocabulary.")
            self.build_vocab()
        else:
            logger.info("Loading an existing vocabulary file.")
            self.load_vocab()

        # build the bag of words file
        if self.rebuild or not os.path.isfile(self.mm_file):
            logger.info("Building a new bag-of-words file.")
            corpus = self._bow_generator()
            corpora.MmCorpus.serialize(fname=self.mm_file, corpus=corpus)
        else:
            logger.info("Loading an existing bag-of-words file.")
            self.load_bow()
        
        self.num_docs = self._get_meta_data()
    
    def _get_meta_data(self):
        """
        look up how many documents are stored in this instance's mm file
        this should be hidden, no need to call directly
        """
        with open(self.mm_file, "r") as fin:
            _ = fin.readline()
            num_docs = int(filter(None, fin.readline().replace("\n", "").split())[0])
        return num_docs

    def build_vocab(self):
        """
        iterate once over the corpus to build the dictionary
        """
        self.vocab = corpora.Dictionary(tokens for tokens in JournalTokens(self.journal_file))
        if self.keep_n is not None:
            self.vocab.filter_extremes(no_below=1, no_above=1.0, keep_n=self.keep_n)
            self.vocab.compactify()

        # save the dictionary for next time
        self.vocab.save(self.vocab_file)
            
    def _bow_generator(self):
        """
        create a generator over journals and use the vocabulary object to convert to a bow
        """
        if self.vocab is None:
            raise ValueError("vocab object not created")
        
        for tokens in JournalTokens(filename=self.journal_file):
            yield self.vocab.doc2bow(tokens)

    def load_bow(self):
        self.bow = corpora.MmCorpus(self.bow_file)

    def load_vocab(self):
        self.vocab = corpora.Dictionary.load(self.vocab_file)
        

def main():
    parser = argparse.ArgumentParser(description='This program shows how to use the Documents class to create bag-of-words and vocabulary files..')
    parser.add_argument('-j', '--journal_file', type=str, help='Full path to the journal file to extract tokens from.')
    parser.add_argument('-d', '--data_dir', type=str, help='Directory of where to save or load bag-of-words vocabulary files.')
    parser.add_argument('-k', '--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--rebuild', dest="rebuild", action="store_true", help='Add this flag to rebuild the bag-of-words and vocabulary, even if copies of the files already exists.')
    parser.set_defaults(verbose=True)
    parser.set_defaults(rebuild=True)
    args = parser.parse_args()
    
    print('documents.py')
    print(args)
    
    start = time.time()
    docs = Documents(journal_file = args.journal_file, data_dir=args.data_dir, rebuild=args.rebuild, keep_n=args.keep_n, verbose=args.verbose)
    docs.fit()
    end = time.time()
    print("Time to convert journal tokens into a BOW and vocabulary: " + str(end - start) + " seconds.")

    
if __name__ == "__main__":
    main()
