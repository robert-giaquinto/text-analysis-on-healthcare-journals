from __future__ import unicode_literals, print_function, absolute_import, division
import spacy
import os
import re

def main():

    data_dir = "/home/srivbane/shared/caringbridge/data/word_embeddings/"
    input_file = "clean_journals_for_word2vec.txt"
    output_file = "clean_sentences_for_word2vec.txt"
    
    # initialize sentence parser
    nlp = spacy.load('en')

    # initialize punctuaction remove
    punct = re.compile(ur"[^a-zA-Z ]", re.UNICODE) # double check this

    print("Beginning parsing...")
    with open(os.path.join(data_dir, input_file), "r") as f, open(os.path.join(data_dir, output_file), "wb") as o:
        for line in f:
            fields = line.replace("\n","").split("\t")
            doc = nlp(fields[-1].replace(u"_", ""))
            s = -1
            for sentence in doc.sents:
                if sentence is None:
                    continue
                else:
                    s += 1
                
                # remove last punctuation from sentence
                cleaned_sentence = re.sub(r"\s+", " ", punct.sub(u" ", sentence.text)).lower()
                o.write("\t".join(fields[0:-1]) + "\t" +  str(s) + "\t" + cleaned_sentence + "\n")

    print("Finished parsing.")

    
if __name__ == "__main__":
    main()
