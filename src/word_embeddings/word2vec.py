from __future__ import division, print_function, absolute_import
import gensim
from src.utilities import pickle_it
import argparse
import numpy as np
import logging
import os

logger = logging.getLogger(__name__)

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
        with open(self.input_file, "r") as f:
            for line in f:
                fields = line.split("\t")
                tokens = fields[-1].split()  # list of tokens
                if len(tokens) > 1:
                    yield tokens


class W2V(object):
    """
    word2vec model.
    """
    def __init__(self, n_jobs, epochs=25, size=100, verbose=True):
        self.n_jobs = n_jobs
        self.epochs = epochs
        self.size = size
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
            size=self.size,  # word vector dimensions
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
    parser.add_argument('--workers', default=1, type=int, help="Number of processes/cores to use.")
    parser.add_argument('--epochs', default=25, type=int, help="Number of epochs to run.")
    parser.add_argument('--size', default=100, type=int, help="Size of word vectors to find.")
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print("word2vec.py")
    print(args)

    w2v_filename = args.data_dir + "w2v_of_" + args.input_file.replace(".txt", ".p")
    w2v = W2V(n_jobs=args.workers, epochs=args.epochs, size=args.size, verbose=args.verbose)
    w2v.fit(args.data_dir + args.input_file)
    pickle_it(w2v, w2v_filename)

    # find words similar to health conditions
    cancer = w2v.model.most_similar(positive=['cancer'], topn=10)
    print("Cancer:", ' '.join([w for w, v in cancer]))
    surgery = w2v.model.most_similar(positive=['surgery', 'transplant'], topn=10)
    print("Surgery:", ' '.join([w for w, v in surgery]))
    injury = w2v.model.most_similar(positive=['injury'], topn=10)
    print("Injury:", ' '.join([w for w, v in injury]))
    stroke = w2v.model.most_similar(positive=['stroke'], topn=10)
    print("Stroke:", ' '.join([w for w, v in stroke]))
    neuro = w2v.model.most_similar(positive=['neurological'], topn=10)
    print("Neurological:", ' '.join([w for w, v in neuro]))
    birth = w2v.model.most_similar(positive=['infant', 'birth'], topn=10)
    print("Childbirth:", ' '.join([w for w, v in birth]))
    congenital = w2v.model.most_similar(positive=['congenital', 'immune'], topn=10)
    print("Congenital:", ' '.join([w for w, v in congenital]))

    
    vocab = set(w2v.model.index2word)
    n = 15
    terms = ['purpose', 'healing', 'heal', 'community', 'support', 'emotional', 'mental', 'need', 'kindness', 'compassion', 'isolation', 'thank', 'hope', 'well', 'being', 'reflect', 'anxious', 'accept', 'aware']
    with open(os.path.join(args.data_dir, 'key_concept_vectors.txt'), 'wb') as f:
        f.write('key_concept\tterm\t' + '\t'.join(['vec' + str(i) for i in range(100)]) + '\n')
        for t in terms:
            similar = w2v.model.most_similar(positive=[t], topn=n)
            if t in vocab:
                vec = w2v.model[t]
            else:
                continue
                
            f.write(t + '\t' + t + '\t' + '\t'.join([str(v) for v in vec]) + '\n')
            print(t + ":")
            for w, v in similar:
                print("\t", w, v)
                vec = w2v.model[w]
                f.write(t + '\t' + w + '\t' + '\t'.join([str(v) for v in vec]) + '\n') 

            
    
if __name__ == "__main__":
    main()
