from __future__ import division, print_function, absolute_import
import gensim
from src.word_embeddings.word2vec import W2V
import argparse
import numpy as np
import cPickle as pickle

def unpickle_it(filepath):
    infile = open(filepath, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj


def main():
    parser = argparse.ArgumentParser(description='Train word2vec models on an input file.')
    parser.add_argument('--w2v_file', type=str, default='/home/srivbane/shared/caringbridge/data/word_embeddings/w2v_of_clean_sentences_for_word2vec.p', help='Name of pickled word2vec model file.')
    args = parser.parse_args()
    print("word2vec.py")
    print(args)

    print("Unpacking model, this may taken a moment...")
    w2v = unpickle_it(args.w2v_file)
    vocab = set(w2v.model.index2word)
    results = []
    positive_seeds = []
    negative_seeds = []

    topn = int(raw_input("How many similar words would you like to show for each input term: "))
    print("Note you can look for words positively (default) or negatively matching a term")
    print("If a term is followed by a dash, for example: happy-")
    print("Then it is considered negatively")


    q = ''
    while q != 'q':
        raw_terms = raw_input("\nPlease enter terms (space separated) that you're interested in: ")
        terms = raw_terms.split()
        positive_terms = []
        negative_terms = []
        for i, pos in enumerate(terms):
            if pos[-1] == "-":
                neg = pos[0:-1]
                if neg in vocab:
                    negative_terms.append(neg)
                else:
                    print("Sorry", neg, "is not in the model's vocabulary.")

            else:
                if pos in vocab:
                    positive_terms.append(pos)
                else:
                    print("Sorry", pos, "is not in the model's vocabulary.")
                
        if len(positive_terms) + len(negative_terms) == 0:
            print("No suitable words entered, trying again...")
            continue
                
        print("You entered the following words for positive similarity:", positive_terms)
        positive_seeds.append(positive_terms)
        print("And the following words for negative similarity:", negative_terms)
        negative_seeds.append(negative_terms)

        res = w2v.model.most_similar(positive=positive_terms, negative=negative_terms, topn=topn)
        results.append(res)
        print("Results:")
        max_len = max([len(w) for w, v in res])
        print("\tWord" + ' '*(max_len+1) + "Similarity")
        for word, vec in res:
            l = len(word)
            print('\t' + word, ' '*(max_len+5-l), round(vec,3))

        q = raw_input("Enter 'q' if you want to stop, otherwise type anything else to lookup more words: ")

    fname = raw_input("Press enter to quit, otherwise all the results (including word vectors) will be written to the file you specify: ")
    if fname != '':
        with open(fname, 'wb') as f:
            f.write('positive_seed\tnegative_seed\tterm\t' + '\t'.join(['vec' + str(i) for i in range(100)]) + '\n')
            for res, p, n in zip(results, positive_seeds, negative_seeds):
                seed_vecs = []
                for i, t in enumerate(p + n):
                    if i == 0:
                        seed_vecs = w2v.model[t]
                    else:
                        new_vec = w2v.model[t]
                        seed_vecs = [s + v for s, v in zip(seed_vecs, new_vec)]
                
                seed_vecs = [s / len(p + n) for s in seed_vecs]
            
                f.write('_'.join(p) + '\t' + '_'.join(n) + '\t' + '_'.join(p+n)  + '\t' + '\t'.join([str(v) for v in seed_vecs]) + '\n')
                for w, v in res:
                    vec = w2v.model[w]
                    f.write('_'.join(p) + '\t' + '_'.join(n) + '\t' + w + '\t' + '\t'.join([str(v) for v in vec]) + '\n')

            
            
if __name__ == "__main__":
    main()
