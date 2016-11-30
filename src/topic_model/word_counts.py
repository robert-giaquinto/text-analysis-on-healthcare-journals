from gensim.corpora import Dictionary, MmCorpus
import os


def main():
    data_dir = '/home/srivbane/shared/caringbridge/data/topic_model_no_names/'
    mm_file = 'train_bow_for_clean_journals_no_names_with_10000_test_docs_25000_terms.mm'
    dict_file = 'vocab_25000.dict'
    output_file = 'word_counts.txt'
    
    bow = MmCorpus(data_dir + mm_file)
    vocab = Dictionary.load(data_dir + dict_file)

    # collect words counts over all documents
    token2freq = {}
    for doc in bow:
        for term_id, f in doc:
            freq = int(f)
            term = vocab[term_id]
            if term not in token2freq:
                token2freq[term] = freq
            else:
                token2freq[term] += freq

    # convert dictionary to a list so we can easily sort by frequency
    tf_pairs = [(term, freq) for term, freq in token2freq.iteritems()]
    tf_pairs = sorted(tf_pairs, key=lambda x: x[1], reverse=True)
    with open(data_dir + output_file, 'wb') as fout:
        for term, freq in tf_pairs:
            fout.write(term + ',' + str(freq) + '\n')
    




if __name__ == "__main__":
    main()
