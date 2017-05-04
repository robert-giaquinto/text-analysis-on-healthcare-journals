from __future__ import division, print_function, absolute_import
import gensim
from src.utilities import pickle_it, unpickle_it
import argparse
import os
import numpy as np
from src.word_embeddings.word2vec import W2V

def compute_site_vectors(input_file, output_file, wv, method="mean"):
    """
    Input file is the parsed sentences file.
    """
    vocab = set(wv.index2word)
    prev_key = "-1"
    with open(input_file, "r") as infile, open(output_file, 'wb') as outfile:
        for i, line in enumerate(infile):
            fields = line.split("\t")
            agg_key = fields[0]

            if agg_key != prev_key:
                # new journal found, score the last text
                if i > 0:
                    try:
                        if method == "mean":
                            vec = np.array([wv[t] for t in tokens if t in vocab]).mean(axis=0)
                        else:
                            vec = np.array([wv[t] for t in tokens if t in vocab]).sum(axis=0)
                    except:
                        print(tokens, vec)
                        
                    if type(vec) != np.ndarray or len(vec) == 0:
                        vec = np.repeat(0.0, 100)
                
                    outfile.write(agg_key + '\t' + '\t'.join([str(v) for v in vec]) + '\n')
                    
                # begin accumulating the new journal
                tokens = fields[-1].split()
            else:
                tokens += fields[-1].split()

            prev_key = agg_key

def main():
    parser = argparse.ArgumentParser(description='Use word2vec model to compute a semantic vector of each document.')
    parser.add_argument('--input_file', type=str, help='Name of input file.')
    parser.add_argument('--output_file', type=str, help='Name of output file.')
    parser.add_argument('--data_dir', type=str, help="Data directory.")
    parser.add_argument('--method', type=str, default='mean', help="Method for aggregating words into site vectors.")
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    print("document_vectors.py")
    print(args)

    # load the word2vec model
    input_file = os.path.join(args.data_dir, args.input_file)
    w2v_file = os.path.join(args.data_dir, 'w2v_of_' + args.input_file.replace('.txt', '.p'))
    output_file = os.path.join(args.data_dir, args.output_file)
    print("loading w2v file:", w2v_file)
    w2v = unpickle_it(w2v_file)
    wv = w2v.model
    del w2v
    compute_site_vectors(input_file, output_file, wv, method=args.method)

if __name__ == "__main__":
    main()
