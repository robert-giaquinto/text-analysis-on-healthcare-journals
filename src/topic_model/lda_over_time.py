from __future__ import division, print_function, absolute_import
import argparse
import os
import logging
import itertools
import numpy as np

from src.topic_model.documents import Documents
from src.topic_model.gensim_lda import GensimLDA
from src.utilities import pickle_it, unpickle_it


# initialize logging
logger = logging.getLogger(__name__)

gensim_logger1 = logging.getLogger('gensim.models.ldamodel')
gensim_logger2 = logging.getLogger('gensim.models.ldamulticore')

# filter out the un-needed  gensim logging messages
class NoGensimFilter(logging.Filter):
    def filter(self, record):
        #useless = record.getMessage().startswith('PROGRESS')  or record.funcName == "blend" or record.funcName == "show_topics"
        useless = record.funcName == "blend" or record.funcName == "show_topics"
        return not useless

gensim_logger1.addFilter(NoGensimFilter())
gensim_logger2.addFilter(NoGensimFilter())


def chunker(iterable, chunksizes):
    it = iter(iterable)
    for chunksize in chunksizes:
        chunk = itertools.islice(it, chunksize)
        if not chunk:
            break
        def inner(c):
            yield (doc for doc in c)

        yield inner(chunk)


def fit_all(train_docs, test_docs, train_bins, test_bins, num_topics, chunksize, passes, n_workers):
    # train phase:
    if chunksize is None:
            chunksize = sum(train_bins)

    m = GensimLDA(docs=train_docs, n_workers=n_workers, verbose=True)
    perf = m.fit(num_topics=num_topics, chunksizes=chunksize, perplexity_threshold=0.1, evals_per_pass=1, max_passes=passes)

    # test phase:
    test_stream = chunker(test_docs.train_bow, test_bins)
    rval = []
    for i, (test_size, test_chunk) in enumerate(zip(test_bins, test_stream)):
        test_chunk = [doc for gen in test_chunk for doc in gen]
        lhood, beta_lhood, corpus_words = m._perword_lhood(m.model, test_chunk)
        var_lhood = lhood / corpus_words
        full_lhood = (beta_lhood + lhood) / corpus_words
        logger.info("var per word lhood[t={}]: {:.2f} / {} = {:.2f}".format(i, lhood, corpus_words, var_lhood))
        logger.info("per word lhood[t={}]: ({:.2f} + {:.2f}) / {} = {:.2f}".format(i, lhood, beta_lhood, corpus_words, full_lhood))
        rval.append((var_lhood, full_lhood))

    # report overall results
    lhood, beta_lhood, corpus_words = m._perword_lhood(m.model, test_docs.train_bow)
    logger.info("var per word lhood: {:.2f} / {} = {:.2f}".format(lhood, corpus_words, lhood / corpus_words))
    logger.info("per word lhood: ({:.2f} + {:.2f}) / {} = {:.2f}".format(lhood, beta_lhood, corpus_words, (beta_lhood + lhood) / corpus_words))
    return rval


def fit_local(train_docs, test_docs, train_bins, test_bins, num_topics, chunksize, passes, n_workers):
    if len(train_bins) != len(test_bins):
        raise ValueError("length of train and test bin counts must be same")
    num_times = len(train_bins)

    # at each time step read in the number of documents specified by the bin sizes for training and testing
    train_docs.load_bow()
    test_docs.load_bow()
    train_bow = [doc for doc in train_docs.train_bow]
    test_bow = [doc for doc in test_docs.train_bow]
    rval = []
    for i in range(1, num_times):
        train_size = sum(train_bins[0:i])
        prev_test_size = sum(test_bins[0:i])
        test_size = test_bins[i]
        logger.info("LDA Local time step: {}. With {} training, and {} testing documents".format(i, train_size, test_size))

        # initialize the mini batch of Documents
        train_chunk = Documents(journal_file=train_docs.journal_file,
                                    num_test=0,
                                    data_dir=train_docs.data_dir,
                                    keep_n=train_docs.keep_n,
                                    no_above=train_docs.no_above,
                                    rebuild=False,
                                    num_docs=train_size,
                                    shuffle=False,
                                    prune_at=train_docs.prune_at,
                                    verbose=True)
        train_chunk.num_train = train_size
        train_chunk.train_bow = train_bow[0:train_size]
        train_chunk.test_bow = []
        train_chunk.vocab = train_docs.vocab

        # train
        if chunksize is None:
            chunksize = train_size
        m = GensimLDA(docs=train_chunk, n_workers=n_workers, verbose=True)
        perf = m.fit(num_topics=num_topics,
                     chunksizes=chunksize,
                     perplexity_threshold=0.1,
                     evals_per_pass=1,
                     max_passes=passes,
                     save_model="t" + str(i) + "_lda_local.lda")

        # test
        test_chunk = test_bow[prev_test_size:(prev_test_size + test_size)]
        lhood, beta_lhood, corpus_words = m._perword_lhood(m.model, test_chunk)
        var_lhood = lhood / corpus_words
        full_lhood = (beta_lhood + lhood) / corpus_words
        logger.info("var per word lhood[t={}]: {:.2f} / {} = {:.2f}".format(i, lhood, corpus_words, var_lhood))
        logger.info("per word lhood[t={}]: ({:.2f} + {:.2f}) / {} = {:.2f}".format(i, lhood, beta_lhood, corpus_words, full_lhood))
        rval.append((var_lhood, full_lhood))

    return rval


def read_bins(bin_file):
    bin_counts = []
    with open(bin_file, "r") as f:
        for i, line in enumerate(f):
            if line == "\n":
                break

            val = int(float(line.replace("\n", "")))
            try:
                if i == 0:
                    continue
                else:
                    bin_counts.append(val)
            except:
                print(line)
    return sum(bin_counts), bin_counts


def main():
    parser = argparse.ArgumentParser(description='Wrapper around the gensim LDA model.')
    parser.add_argument('--train', type=str, help='Training file.')
    parser.add_argument('--test', type=str, help='Testing file.')
    parser.add_argument('--train_bins', type=str, help='File showing how many documents are in each of the time step bins for training set.')
    parser.add_argument('--test_bins', type=str, help='File showing how many documents are in each of the time step bins for training set.')
    parser.add_argument('--data_dir', type=str, default="/home/srivbane/shared/caringbridge/data/clean_journals/", help='Directory of where to save or load bag-of-words, vocabulary, and model performance files.')
    parser.add_argument('--fit_all', dest='fit_all', action='store_true', help='Should this fit LDA to all time steps, or each time step individually?')

    parser.add_argument('--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--no_above', type=float, default=0.9, help='Drop any terms appearing in at least this proportion of the documents.')
    parser.add_argument('--num_topics', type=int, default=25, help="Number of topics to extract. Multiple arguments can be given to test a range of parameters.")

    parser.add_argument('--n_workers', type=int, default=1, help="Number of cores to run on.")
    parser.add_argument('--chunksize', type=int, default=None, help="Mini-batch size for model training.")
    parser.add_argument('--passes', type=int, default=1, help="Number of passes over the corpus for every training instance.")

    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--rebuild', dest="rebuild", action="store_true", help='Add this flag to rebuild the bag-of-words and vocabulary, even if copies of the files already exists.')
    parser.add_argument('--data', dest="data", action="store_true", help='Only build the data needed for these analyses and then stop.')
    parser.set_defaults(verbose=False)
    parser.set_defaults(rebuild=False)
    parser.set_defaults(fit_all=False)
    parser.set_defaults(data=False)
    args = parser.parse_args()

    print('lda_over_time.py')
    print(args)

    # read in counts of documents at each time step
    total_train, train_bins = read_bins(args.train_bins)
    total_test, test_bins = read_bins(args.test_bins)

    # create document objects
    print("Creating training documents object")
    train_docs = Documents(journal_file=args.train,
                           num_test=0,
                           data_dir=args.data_dir,
                           rebuild=args.rebuild,
                           keep_n=args.keep_n,
                           no_above=args.no_above,
                           num_docs=total_train,
                           shuffle=False,
                           prune_at=8000000,
                           verbose=args.verbose)
    train_docs.fit()

    print("Creating testing documents")
    test_docs = Documents(journal_file=args.test,
                          num_test=0,
                          data_dir=args.data_dir,
                          rebuild=False,
                          keep_n=args.keep_n,
                          no_above=args.no_above,
                          num_docs=total_test,
                          shuffle=False,
                          prune_at=8000000,
                          verbose=args.verbose)
    test_docs.load_vocab()
    test_docs.num_train = total_test
    if args.rebuild or args.data:
        test_docs.build_bow()

    # in either case load iterator
    test_docs.load_bow()

    if args.data:
        return

    # call the training method
    if args.fit_all:
        print("Training LDA on all time steps")
        results = fit_all(train_docs=train_docs, test_docs=test_docs,
            train_bins=train_bins, test_bins=test_bins,
            num_topics=args.num_topics, chunksize=args.chunksize, passes=args.passes, n_workers=args.n_workers)
    else:
        print("training LDA locally")
        results = fit_local(train_docs=train_docs, test_docs=test_docs,
            train_bins=train_bins, test_bins=test_bins,
            num_topics=args.num_topics, chunksize=args.chunksize, passes=args.passes, n_workers=args.n_workers)

    print("Perplexities over time steps:\n", results)
    fit_type = "all" if args.fit_all else "local"
    with open(os.path.join(args.data_dir, "perplexities_all_" + fit_type + ".txt"), "wb") as f:
        f.write("time\tvar_lhood\tfull_lhood\n")
        for t, (var_lhood, full_lhood) in enumerate(results):
            f.write("t\t{:.2f}\t{:.2f}\n".format(t, var_lhood,full_lhood))


if __name__ == "__main__":
    main()
