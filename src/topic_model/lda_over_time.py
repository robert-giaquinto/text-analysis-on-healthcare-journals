from __future__ import division, print_function, absolute_import
import argparse
import os
import logging
import itertools

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
    m = GensimLDA(docs=train_docs, n_workers=n_workers, verbose=True)
    perf = m.fit(num_topics=num_topics, chunksizes=chunksize, perplexity_threshold=0.0, evals_per_pass=None, max_passes=passes)

    # test phase:
    test_stream = chunker(test_docs.train_bow, test_bins)
    perplexities = []
    for test_size, test_chunk in zip(test_bins, test_stream):
        perplexity = m._perplexity_score(m.model, (a for b in test_chunk for a in b), total_docs=train_docs.num_train, num_sample=test_size)
        perplexities.append(perplexity)

    return perplexities



def fit_local(train_docs, test_docs, train_bins, test_bins, num_topics, chunksize, passes, n_workers):
    # at each time step read in the number of documents specified by the bin sizes for training and testing
    train_docs.load_bow()
    test_docs.load_bow()
    train_stream = chunker(train_docs.train_bow, train_bins)
    test_stream = chunker(test_docs.train_bow, test_bins)
    perplexities = []
    for train_size, test_size, train_chunk, test_chunk in zip(train_bins, test_bins, train_stream, test_stream):
        # initialize the mini batch of Documents
        train_doc_chunk = Documents(journal_file=train_docs.journal_file,
                                    num_test=0,
                                    data_dir=train_docs.data_dir,
                                    keep_n=train_docs.keep_n,
                                    no_above=train_docs.no_above,
                                    rebuild=False,
                                    num_docs=train_size,
                                    shuffle=False,
                                    prune_at=train_docs.prune_at,
                                    verbose=True)
        train_doc_chunk.num_train = train_size
        train_doc_chunk.train_bow = train_chunk
        train_doc_chunk.vocab = train_docs.vocab
        # train
        m = GensimLDA(docs=train_doc_chunk, n_workers=n_workers, verbose=True)
        perf = m.fit(num_topics=num_topics, chunksizes=chunksize, perplexity_threshold=0.0, evals_per_pass=None, max_passes=passes)
        # test
        perplexity = m._perplexity_score(m.model, (a for b in test_chunk for a in b), total_docs=train_size, num_sample=test_size)
        perplexities.append(perplexity)
        
    return perplexities



def read_bins(bin_file):
    total = 0
    bin_counts = []
    with open(bin_file, "rb") as f:
        for i, line in enumerate(f):
            if line == "\n":
                break
            
            fields = line.split('\t')
            try:
                if i == 0:
                    total = int(fields[1])
                else:
                    bin_counts.append(int(fields[1]))
            except:
                print(line)
    return total, bin_counts


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
    parser.add_argument('--chunksize', type=int, default=1024, help="Mini-batch size for model training.")
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
        perplexities = fit_all(train_docs, test_docs, train_bins, test_bins, args.num_topics, args.chunksize, args.passes, args.n_workers)
    else:
        print("training LDA locally")
        perplexities = fit_local(train_docs, test_docs, train_bins, test_bins, args.num_topics, args.chunksize, args.passes, args.n_workers)

    print("Perplexities over time steps:\n", perplexities)
    with open(os.path.join(args.data_dir, "perplexities_all_" + str(args.fit_all) + ".txt"), "wb") as f:
        for p in perplexities:
            f.write(str(p) + "\n")
    
    
if __name__ == "__main__":
    main()
