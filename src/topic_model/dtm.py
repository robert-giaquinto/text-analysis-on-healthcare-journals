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
    parser.add_argument('--holdout_file', type=str, help='Holdout set.')
    parser.add_argument('--data_dir', type=str, help='Directory of where to save or load bag-of-words, vocabulary, and model performance files.')
    parser.add_argument('--dtm_binary', type=str, help='Path for where to find the dynamic topic model binary file.')
    parser.add_argument('--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--num_train', type=int, help="Number of documents in the journal file (specifying this can speed things up).")
    parser.add_argument('--num_test', type=int, help="Number of documents in the holdout set.")
    parser.add_argument('--num_topics', type=int, help="Number of topics in the model you want to use.")
    args = parser.parse_args()

    print('dtm.py')
    print(args)

    print("Loading trianing documents")
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

    print("Loading a holdout set")
    holdout = Documents(journal_file=args.holdout_file,
                        num_test=0,
                        data_dir=args.data_dir,
                        rebuild=False,
                        keep_n=args.keep_n,
                        num_docs=args.num_test,
                        shuffle=False,
                        verbose=True)
    holdout.fit()
    holdout.load_bow()
    
    print("convert corpus to the LDA-C format needed")
    holdout_file = args.data_dir + "holdout-mult.dat"
    train_file = args.data_dir + "train-mult.dat"
    if not os.path.isfile(train_file):
        corpora.BleiCorpus.save_corpus(holdout_file, holdout.train_bow, id2word=docs.vocab)
        corpora.BleiCorpus.save_corpus(train_file, docs.train_bow, id2word=docs.vocab)
    

    os.chdir("/home/srivbane/shared/caringbridge/data/dtm/")
    cmd = "/home/srivbane/smit7982/dtm/dtm/main --ntopics={p1} --model=dtm --mode=fit --initialize_lda=true --corpus_prefix=train --outname=train_out".format(p1=args.num_topics)
    cmd += " --alpha=0.01  --lda_max_em_iter=5 --lda_sequence_min_iter=1 --lda_sequence_max_iter=3 --top_chain_var=0.01 --rng_seed=0"
    print("Running command", cmd)
    subprocess.call(cmd, shell=True)
    print("training complete")


    print("scoring holdout set")
    cmd = "/home/srivbane/smit7982/dtm/dtm/main --ntopics={p1} --mode=time --corpus_prefix=train --rng_seed=0".format(p1=args.num_topics)
    cmd += " --heldout_corpus_prefix=holdout --lda_model_prefix=train_out/lda-seq/ --outname=out"
    subprocess.call(cmd, shell=True)
    
if __name__ == '__main__':
    main()
