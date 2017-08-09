from __future__ import division, print_function, absolute_import
import os
import argparse
import logging
from gensim import corpora, utils
from gensim import utils, corpora, matutils
import subprocess

from src.topic_model.documents import Documents

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description='Wrapper around the gensim LDA model.')
    parser.add_argument('--train_file', type=str, help='Training set.')
    parser.add_argument('--test_file', type=str, help='Test set.')
    parser.add_argument('--data_dir', type=str, help='Directory of where to save or load bag-of-words, vocabulary, and model performance files.')
    parser.add_argument('--dtm_binary', type=str, help='Path for where to find the dynamic topic model binary file.')
    parser.add_argument('--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--num_train', type=int, default=None, help="Number of documents in the journal file (specifying this can speed things up).")
    parser.add_argument('--num_test', type=int, default=None, help="Number of documents in the test set.")
    parser.add_argument('--num_topics', type=int, help="Number of topics in the model you want to use.")
    parser.add_argument('--data', dest="data", action="store_true", help='Only build the data needed for these analyses and then stop.')
    parser.set_defaults(data=False)
    args = parser.parse_args()

    print('dtm.py')
    print(args)

    test_file = args.data_dir + "test-mult.dat"
    train_file = args.data_dir + "train-mult.dat"
    if not os.path.isfile(train_file):
        print("Loading training documents")
        docs = Documents(journal_file=args.train_file,
                         num_test=0,
                         data_dir=args.data_dir,
                         rebuild=False,
                         keep_n=args.keep_n,
                         num_docs=args.num_train,
                         shuffle=False,
                         verbose=True)
    
        docs.fit()
        docs.load_vocab()
        docs.load_bow()

        print("Loading a test set")
        test = Documents(journal_file=args.test_file,
                            num_test=0,
                            data_dir=args.data_dir,
                            rebuild=False,
                            keep_n=args.keep_n,
                            num_docs=args.num_test,
                            shuffle=False,
                            verbose=True)
        test.fit()
        test.load_bow()
        
        
        print("Need to convert corpus to the LDA-C format")
        corpora.BleiCorpus.save_corpus(test_file, test.train_bow, id2word=docs.vocab)
        corpora.BleiCorpus.save_corpus(train_file, docs.train_bow, id2word=docs.vocab)
    

    if not args.data:
        os.chdir(args.data_dir)
        cmd = "/home/srivbane/smit7982/dtm/dtm/main --ntopics={p1} --model=dtm --mode=fit --initialize_lda=true --corpus_prefix={p2} --outname={p3}".format(p1=args.num_topics, p2=args.data_dir + "train-authors", p3=args.data_dir + "dtm_out")
        cmd += " --alpha=0.01  --lda_max_em_iter=20 --lda_sequence_min_iter=6 --lda_sequence_max_iter=20 --top_chain_var=0.005 --rng_seed=0"
        print("Running command", cmd)
        subprocess.call(cmd, shell=True)
        print("training complete")

        print("scoring test set")
        cmd = "/home/srivbane/smit7982/dtm/dtm/main --ntopics={p1} --mode=time --corpus_prefix={p2} --rng_seed=0".format(p1=args.num_topics, p2=args.data_dir + "train-authors")
        cmd += " --heldout_corpus_prefix={p3} --lda_model_prefix={p4} --outname={p5}".format(p3=args.data_dir + "test-authors", p4=args.data_dir + "dtm_out/lda-seq/", p5=args.data_dir + "dtm_inf_out")
        print("Running command", cmd)
        subprocess.call(cmd, shell=True)
    
if __name__ == '__main__':
    main()
