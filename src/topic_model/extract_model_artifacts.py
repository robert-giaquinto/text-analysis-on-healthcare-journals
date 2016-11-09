from __future__ import division, print_function, absolute_import
import os

from src.utilities import unpickle_it
from src.topic_model.gensim_lda import GensimLDA
from src.topic_model.documents import Documents


def main():
    data_dir = '/home/srivbane/shared/caringbridge/data/topic_model/'
    lda_file = 'LDA_test_10000_train_13747900_topics_100.p'
    print("Unpickling model")
    model = unpickle_it(data_dir + lda_file)
    model.topic_term_method = 'unweighted'

    print("Saving word topic probabilities")
    model.save_word_topic_probs(os.path.join(data_dir, "unranked_word_topic_probs.txt"), metric="none")
    print("Saving topic terms")
    model.save_topic_terms(os.path.join(data_dir, "unranked_unweighted_topic_terms.txt"), metric="none")

    save_doc_topics = False
    if save_doc_topics:
        docs = Documents(journal_file=data_dir + 'clean_journals/clean_journals_for_topic.txt',
                         num_test=10000,
                         data_dir=data_dir + 'topic_model',
                         rebuild=False,
                         keep_n=25000,
                         num_docs=13757900,
                         verbose=True)
        model._init_docs(docs)    
        model.save_doc_topic_probs(docs.train_bow, docs.train_keys, os.path.join(args.data_dir, "train_document_topic_probs.txt"))
    

if __name__ == '__main__':
    main()
