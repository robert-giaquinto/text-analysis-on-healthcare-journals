from __future__ import division, print_function, absolute_import
import os
from gensim import corpora
import time
import argparse
import logging

from src.utilities import count_lines, shuffle_file, split_file
from src.topic_model.journal_tokens import JournalTokens
from src.topic_model.journal_keys import JournalKeys

logger = logging.getLogger(__name__)


class Documents(object):
    """
    Iterate over the cleaned journal tokens to create a bag-of-words
    dataset, and a vocabulary

    Implementing this as a class to keep track of all the variables
    used creating the vocabulary, and storing/loading the BOW data
    without needed to rebuild the BOW every time
    """
    def __init__(self, journal_file, num_test, data_dir=None, keep_n=25000, rebuild=True, num_docs=None, shuffle=True, verbose=False):
        """
        Args:
            journal_file: full file name of the big flat file of all tokenized journals and their keys.
            num_test: How many of the documents to hold out in a test set for evaluation
            data_dir: folder to save the output vocab and bow files.
            keep_n: number of terms to keep in the vocabulary.
            rebuild: true to rewrite the output files, false to use existing files (if they exist)
            num_docs: Number of documents in the journal file. Specifying this can speed things up, if you know it.
            verbose: print progress to log
        Return: None
        """
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.journal_file = journal_file
        self.keep_n = keep_n
        self.rebuild = rebuild
        self.num_docs = num_docs
        self.num_train = None
        self.num_test = num_test
        self.shuffle = shuffle

        if data_dir is None:
            self.data_dir = os.path.dirname(journal_file).replace("clean_journals", "topic_model")
        else:
            self.data_dir = data_dir

        # init variable to store the vocab in memory
        self.vocab = None
        self.vocab_file = os.path.join(self.data_dir, "vocab_" + str(keep_n) + ".dict")
        # init generator over the bag of words files for training and testing
        self.train_bow = None
        self.test_bow = None
        # init keys generator corresponding to BOW file for training and test
        prefix = os.path.splitext(os.path.split(journal_file)[-1])[0]
        self.train_shard_file = os.path.join(os.path.dirname(journal_file), prefix + "_shards", prefix + "_01_of_2.txt")
        self.train_keys = None
        self.test_shard_file = os.path.join(os.path.dirname(journal_file), prefix + "_shards", prefix + "_02_of_2.txt")
        self.test_keys = None
        # name of file where MatrixMarket bag-of-words data is stored
        self.train_file = os.path.join(self.data_dir, "train_bow_for_" + prefix + "_with_" + str(num_test) + "_test_docs_" + str(keep_n) + "_terms.mm")
        if num_test > 0:
            self.test_file = os.path.join(self.data_dir, "test_bow_for_" + prefix + "_with_" + str(num_test) + "_test_docs_" + str(keep_n) + "_terms.mm")
        else:
            self.test_file = None

    def fit(self):
        """
        iterate over the journal file and create the BOW of file
        and the vocabulary

        Note: building the vocabulary over the entire corpus -- we're not
        too concerned with how well the algorithm generalizes to new data/words.
        Just want to be able to measure how well the model fits held out data
        during training.
        """
        if self.rebuild or not os.path.isfile(self.vocab_file):
            # build the vocabulary file
            logger.info("Building a new vocabulary.")
            self.build_vocab()
        else:
            logger.info("Loading an existing vocabulary file.")
            self.load_vocab()


        if self.num_docs is None:
            self.num_docs = count_lines(self.journal_file)
            logger.info("Counted " + str(self.num_docs) + " documents in journal file.")

        if self.num_docs <= self.num_test:
            raise ValueError("You must allocate at least one document to be in the test set")

        self.num_train = self.num_docs - self.num_test

        # build the bag of words file
        if self.rebuild or not os.path.isfile(self.train_file):
            logger.info("Building a new bag-of-words file.")
            self.build_bow()
        else:
            logger.info("Loading an existing bag-of-words file: %s", self.train_file)
            self.load_bow()

    def build_vocab(self):
        """
        iterate once over the corpus to build the dictionary
        """
        self.vocab = corpora.Dictionary(tokens for tokens in JournalTokens(self.journal_file))
        if self.keep_n is not None:
            # a term cannot appear in more than 90% of docs and only keep the top keep_n terms (remaining)
            self.vocab.filter_extremes(no_below=1, no_above=0.75, keep_n=self.keep_n)
            self.vocab.compactify()

        # save the dictionary for next time
        self.vocab.save(self.vocab_file)

    def build_bow(self):
        if self.num_test > 0:
            logger.info("Splitting into " + str(self.num_train) + " documents for training, and " + str(self.num_test) + " for testing.")

            if self.shuffle:
                logger.info("Randomly shuffling input dataset...")
                shuffle_file(self.journal_file, memory_avail="50%")

            logger.info("Splitting the file into train and test")
            # this is a hacky way to do this (split_file wasn't intended to be used this way), but should work
            self.train_shard_file, self.test_shard_file = split_file(self.journal_file,
                2,
                infile_len=self.num_train * 2)

            test_corpus = self._bow_generator(self.test_shard_file)
            corpora.MmCorpus.serialize(fname=self.test_file, corpus=test_corpus)
        else:
            self.train_shard_file = self.journal_file

        train_corpus = self._bow_generator(self.train_shard_file)
        corpora.MmCorpus.serialize(fname=self.train_file, corpus=train_corpus)


    def _bow_generator(self, filename):
        """
        create a generator over journals and use the vocabulary object to convert to a bow
        """
        if self.vocab is None:
            raise ValueError("vocab object not created")

        for tokens in JournalTokens(filename=filename):
            yield self.vocab.doc2bow(tokens)

    def load_keys(self):
        """
        Load the keys associated with each element in BOW generators
        """
        self.train_keys = (keys for keys in JournalKeys(filename=self.train_shard_file))
        if self.test_shard_file is not None:
            self.test_keys = (keys for keys in JournalKeys(filename=self.test_shard_file))

    def load_bow(self):
        """
        Load the bag-of-words files (stored as a generator)
        """
        self.train_bow = corpora.MmCorpus(self.train_file)
        if self.num_test > 0:
            self.test_bow = corpora.MmCorpus(self.test_file)

    def load_vocab(self):
        """
        Load the gensim Dictionary
        """
        self.vocab = corpora.Dictionary.load(self.vocab_file)


def main():
    parser = argparse.ArgumentParser(description='This program shows how to use the Documents class to create bag-of-words and vocabulary files..')
    parser.add_argument('-j', '--journal_file', type=str, help='Full path to the journal file to extract tokens from.')
    parser.add_argument('-d', '--data_dir', type=str, help='Directory of where to save or load bag-of-words vocabulary files.')
    parser.add_argument('-k', '--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--num_test', type=int, default=0, help="Number of documents to hold out for the test set.")
    parser.add_argument('--num_docs', type=int, help="Number of documents in the journal file (specifying this can speed things up).")
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--rebuild', dest="rebuild", action="store_true", help='Add this flag to rebuild the bag-of-words and vocabulary, even if copies of the files already exists.')
    parser.add_argument('--no_shuffle', dest="shuffle", action="store_false", help='Add this flag to not shuffle the input data before splitting it into training and test BOW files. Shuffling is highly recommended, but can take a time for large datasets.')
    parser.set_defaults(verbose=False)
    parser.set_defaults(rebuild=False)
    parser.set_defaults(shuffle=True)
    args = parser.parse_args()

    print('documents.py')
    print(args)

    start = time.time()
    docs = Documents(journal_file = args.journal_file,
                     num_test=args.num_test,
                     data_dir=args.data_dir,
                     rebuild=args.rebuild,
                     keep_n=args.keep_n,
                     num_docs=args.num_docs,
                     shuffle=args.shuffle, verbose=args.verbose)
    docs.fit()
    end = time.time()
    print("Time to convert journal tokens into a BOW and vocabulary: " + str(end - start) + " seconds.")


if __name__ == "__main__":
    main()
