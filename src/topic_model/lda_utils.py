from __future__ import division, print_function, absolute_import
import os
import subprocess
import numpy as np

from src.topic_model.documents import Documents
from src.topic_model.gensim_lda import GensimLDA
from src.utilities import unpickle_it


def kl_div(p):
    """
    KL Divergence between a distribution (vector) p and a uniform vector
    
    Args:
        p: numpy vector (should sum to 1)
    Returns: KL divergence of p from a uniform distribution
    """
    # uniform distribution
    q = 1.0 * np.ones(len(p)) / len(p)
    kl = np.sum( p * np.log2(p / q) )
    return kl


def compute_doc_topic_stats(infile, outfile):
    """
    Read a file containing topic proportions of each document and
    output a new file with document keys and statistics for extracting
    documents matching certain topics.
    """
    with open(infile,'r') as fin, open(outfile, 'wb') as fout:
        for line in fin:
            fields = line.replace("\n", "").split(',')
            keys = fields[0:4]
            topic_dist = np.array([float(f) for f in fields[4:]])
            max_topic = np.argmax(topic_dist)
            kl = kl_div(topic_dist)
            fout.write(','.join(keys) + ',' + str(max_topic) + ',' + str(kl) + "\n")

            
def select_best_docs_per_topic(stat_file, num_docs):
    """
    For each topic select the num_docs
    documents that most exhibit that topic.

    Args:
        stat_file: comma-delim file containing each documents keys, the kl-div
                   of that documents topics compared to a uniform distribution,
                   and the topic in highest proportion to a specific topic
        num_docs:  Select top num_docs documents that match a specific topic.

    Returns: A list of topic, key pairs
    """
    # sort the stat_file by topic number and then kl-divergence
    cmd = """/bin/bash -c "sort -rn %s -t $',' -k5,6 -o %s -S %s" """ % (stat_file, stat_file, "50%")
    subprocess.call(cmd, shell=True)

    # loop through the file grabbing the top document keys for each topic
    topic_dict = {}    
    prev_topic = -1
    with open(stat_file, 'r') as fin:
        for line in fin:
            fields = line.replace("\n", "").split(",")
            curr_topic = fields[4]
            keys = fields[0:4]

            # general case: want to collect for all topics
            if curr_topic != prev_topic:
                # new topic found, initialize the element in the dict
                topic_dict[curr_topic] = [keys]
                prev_topic = curr_topic
            elif len(topic_dict[curr_topic]) < num_docs:
                # still on same topic, have we selected num_docs yet?
                topic_dict[curr_topic].append(keys)
            else:
                prev_topic = curr_topic
                continue

    # transform the dictionary into a list of tuples (easier to work with)
    rval = []
    for topic, arr in topic_dict.iteritems():
        for keys in arr:
            rval.append((topic, keys))
    return rval


def extract_docs(keys, parsed_file):
    """
    Pull out the journal text for a list of keys

    Get original text stored in parsed_journals

    Assumes order of keys matches order of journals in parsed_file (i.e. they're
    both sorted).
    """
    rval = []
    keys_ptr = 0
    with open(parsed_file, 'r') as fin:
        for line in fin:
            if line.startswith('\t'.join(keys[keys_ptr])):
                # document found!
                fields = line.replace("\n").split("\t")
                rval.append(fields[-1])
                # increment keys pointer so that we know which journal keys to look for next
                keys_ptr += 1
    return rval
    

def extract_best_docs_per_topic(topic_keys, parsed_file):
    """
    Pull out the original journal messages corresponding to a set of journal keys

    This implements extract_doc for each of the topics in the keys dictionary given

    Args:
        keys: assuming this a list of tuples, first elt in tuple is topic, second is a list of journal keys
    Return: 
    """
    # sort the keys so that they'll have same order as the parsed_journal_all.txt file
    topic_keys = sorted(topic_keys, key=lambda x: ''.join(x[1]))

    # split keys into key and values
    topics, keys = zip(*topic_keys)

    # get list of documents
    docs = extract_docs(keys, parsed_file)

    # combine the topics docs for each topic
    rval = zip(topics, docs)
    return rval


def main():
    data_dir = '/home/srivbane/shared/caringbridge/data/dev/topic_model/'
    lda_file = 'LDA_test_100_train_728_topics_10.p'
    parsed_journals_file = '/home/srivbane/shared/caringbridge/data/dev/parsed_json2/parsed_journal_all.txt'
    doc_top_file = None
    outfile = data_dir + 'top_docs_for_each_topic.txt'

    # load lda model
    lda = unpickle_it(os.path.join(data_dir, lda_file))
    lda.load_bow()
    lda.load_vocab()

    # create the document topic file if needed
    if doc_top_file is None:
        doc_top_file = os.path.join(data_dir, "train_document_topic_probs.txt")
        lda.save_doc_topic_probs(lda.docs.train_bow, lda.docs.train_keys, doc_top_file)
        
    # find the best documents for each topic
    topic_keys = select_best_docs_per_topic(doc_top_file, 3)

    # extract these documents
    doc_topics = extract_best_docs_per_topic(topic_keys, parsed_journals_file)

    # write out results to a file
    with open(outfile, 'wb') as fout:
        for topic, doc in doc_topics:
            fout.write(topic + "\t" + doc + "\n")


if __name__ == "__main__":
    main()
