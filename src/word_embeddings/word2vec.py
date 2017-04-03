from __future__ import division, print_function
import gensim
from src.utilities import pickle_it



class SentenceIterator(object):
    """
    This is simply a wrapper over the sentence file to iterate over the tokens (one sentence per line)
    """
    def __init__(self, input_file):
        self.input_file = input_file

    def __iter__(self):
        """
        Output file has fields: headnote_key, key_number, topic_id, case+topic id headnote_key space separated words
        Note all keys are returned/treated as text
        :return:
        """
        for line in open(self.input_file, "r"):
            fields = line.split("\t")
            tokens = fields[-1].split()  # list of tokens
            yield tokens


class W2V(object):
    """
    word2vec model.
    """
    def __init__(self, n_jobs, epochs=25, verbose=True):
        self.n_jobs = n_jobs
        self.epochs = epochs
        self.verbose = verbose
        self.model = None

    def fit(self, input_file):
        if self.verbose:
            print("Creating corpus generator from the input file")

        sentences = SentenceIterator(input_file)

        # run word2vec
        if self.verbose:
            print("Fitting model")

        self.model = gensim.models.Word2Vec(sentences,
            sg=1,  # skip-gram model
            size=200,  # word vector dimensions
            window=5,  # max context window skip length
            hs=1,  # hierarchical softmax enabled
            negative=0,  # no negative sampling
            min_count=5,  # words must occur 5+ times to be in vocabulary
            iter=self.epochs,
            workers=self.n_jobs)

        self.model.init_sims(replace=True) # compress w2v

    def cos_similarity(self, wvec1, wvec2):
        if self.model is None:
            raise ValueError("Must fit model first.")

        return self.model.n_similarity(wvec1, wvec2)


def main():
    parser = argparse.ArgumentParser(description='Train word2vec models on an input file.')
    parser.add_argument('--input_file', type=str, help='Name of input file.')
    parser.add_argument('--data_dir', type=str, help="Data directory.")
    parser.add_argument('--verbose', default=True, type=float, help="Report on progress.")
    parser.add_argument('--workers', default=1, type=int, help="Number of processes/cores to use.")
    parser.add_argument('--epochs', default=25, type=int, help="Number of epochs to run.")
    args = parser.parse_args()

    w2v_filename = args.dir + "w2v_of_" + args.input_file.replace(".txt", ".p")
    w2v = W2V(n_jobs=args.workers, epochs=args.epochs, verbose=args.verbose)
    w2v.fit(args.data_dir + args.input_file)
    pickle_it(w2v, w2v_filename)

if __name__ == "__main__":
    main()
