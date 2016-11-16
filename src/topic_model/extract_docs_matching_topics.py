from __future__ import division, print_function, absolute_import
import os
import subprocess
import re
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

            
class FixedArray(object):
    """
    Structure for holding the documents with the three
    highest topic proportions for each topic.
    
    Works in a streaming fashion, for each new document
    pass it to the add method and the method will determine if
    the object should update which documents it is holding for each topic.
    """
    def __init__(self, num_docs, num_topics):
        self.num_docs = num_docs
        self.num_topics = num_topics
        # store the max topic proportion assuming you want to keep the top num_docs for each topic
        self.topic_maxs = [[-1.0 * float("inf") for d in range(num_docs)] for t in range(num_topics)]
        # save the keys to the corresponding document that is one of the top num_docs matches to each topics
        self.keys = [[[] for d in range(num_docs)] for t in range(num_topics)]

    def add(self, topics, keys):
        for t in range(num_topics):
            for d in range(num_docs):
                if topics[t] > self.topic_maxs[t][d]:
                    self.topic_maxs[t] = self.topic_maxs[0:d] + [topics[t]] + self.topic_maxs[d:-1]
                    self.keys[t] = self.keys[0:d] + keys + self.keys[d:-1]
                    break



def select_docs_per_topic_by_max(infile, num_docs=1):
    with open(infile, 'r') as fin:
        for i, line in enumerate(fin):
            fields = line.replace('\n', '').split(',')
            keys = fields[0:4]
            topic_dist = [float(f) for f in fields[4:]]

            # initialize the list of current max found for each topic
            if i == 0:
                num_topics = len(topic_dist)
                farr = FixedArray(num_docs=num_docs, num_topics=num_topics)

            farr.add(topic_dist, keys)

    # convert the farr object to a list of tuples
    keys_to_max = [(str(t), k) for t, key_arr in enumerate(self.keys) for k in key_arr]
    return keys_to_max
                    
    
            
def select_docs_per_topic_by_kl(stat_file, num_docs):
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
    prev_topic = '-1'
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


def sanitize_docs(doc):
    """
    Perform some basic cleaning of the text to make it more easily readable

    This is meant to be run on the documents that are extracted, so that
    they're presentable to a general audience.

    For now it should suffice to simply remove html formatting (if it exists)
    """
    rval = re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", doc))
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
            if keys_ptr >= len(keys):
                break
            
            if line.startswith('\t'.join(keys[keys_ptr])):
                # document found!
                fields = line.replace("\n", "").split("\t")
                doc = sanitize_docs(fields[-1])
                rval.append(doc)
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
    data_dir = '/home/srivbane/shared/caringbridge/data/topic_model/'
    lda_file = 'LDA_test_10000_train_13747900_topics_100.p'
    parsed_journals_file = '/home/srivbane/shared/caringbridge/data/parsed_json/parsed_journal_all.txt'
    doc_top_file = data_dir + 'train_document_topic_probs.txt'
    stats_file = data_dir + 'kl_divergence_stats.txt'
    outfile = data_dir + 'top_doc_for_each_topic.txt'

    # load lda model
    print("Unpickling model")
    lda = unpickle_it(os.path.join(data_dir, lda_file))
    lda.docs.load_bow()
    lda.docs.load_vocab()
    lda.docs.load_keys()
    
    # create the document topic file if needed
    if doc_top_file is None:
        print("Building a new document topic matrix")
        doc_top_file = os.path.join(data_dir, "train_document_topic_probs.txt")
        lda.save_doc_topic_probs(lda.docs.train_bow, lda.docs.train_keys, doc_top_file)
        
    # find the best documents for each topic
    print("Computing KL stats for each doc")
    compute_doc_topic_stats(doc_top_file, stats_file)
    topic_keys = select_docs_per_topic_by_kl(stats_file, 3)
    print("Selecting best doc matches via max")
    max_topic_keys = select_docs_per_topic_by_max(doc_top_file)
    
    # extract these documents
    print("Extracting docs for each topic")
    doc_topics = extract_best_docs_per_topic(topic_keys, parsed_journals_file)
    print(doc_topics[0])

    print("Extracting docs for each topic via max approach")
    max_doc_topics = extract_best_docs_per_topic(max_topic_keys, parsed_journals_file)
    print(max_doc_topics[0])

    # write out results to a file
    with open(outfile, 'wb') as fout:
        for topic, doc in doc_topics:
            fout.write(topic + "\t" + doc + "\n")

    with open(outfile.replace("top_docs", "max_top_docs"), 'wb') as fout:
        for topic, doc in max_doc_topics:
            fout.write(topic + "\t" + doc + "\n")


if __name__ == "__main__":
    main()
