from __future__ import unicode_literals, print_function, absolute_import, division
import spacy
import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='parse sentences from journals for training word2vec.')
    parser.add_argument('--data_dir', type=str, help='Data directory where input and output files should be/go.')
    parser.add_argument('--input_file', type=str, help='Name of file to read input journals from.')
    parser.add_argument('--output_file', type=str, help='Name of file to write out all the sentences to.')
    args = parser.parse_args()

    print('parse_sentences.py')
    print(args)

    input_file = os.path.join(args.data_dir, args.input_file)
    output_file = os.path.join(args.data_dir, args.output_file)

    # initialize sentence parser
    nlp = spacy.load('en')

    # initialize regexs
    punct = re.compile(ur"[^a-zA-Z0-9_ ]", re.UNICODE)
    years = re.compile(ur"\b(18|19|20)[0-9]{2}\b")
    dollars = re.compile(ur"\b\$[1-9][0-9,\.]*")
    percents = re.compile(ur"[1-9][0-9\.]*\%")
    times = re.compile(ur"(2[0-3]|[01]?[0-9]):([0-5]?[0-9])")
    urls = re.compile(ur'((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?', re.IGNORECASE)
    dates = re.compile(ur"\b[0-3]?[0-9]?[-/][0-3]?[0-9]?[-/](18|19|20)?[0-9]{2}\b")
    emails = re.compile(ur"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b", re.IGNORECASE)
    
    print("Beginning parsing...")
    with open(input_file, "r") as f, open(output_file, "wb") as o:
        for line in f:
            fields = line.replace("\n","").split("\t")
            doc = nlp(fields[-1])
            s = -1
            for sentence in doc.sents:
                if sentence is None:
                    # ignore empty sentences
                    continue
                elif sum((1 if t.is_alpha else 0 for t in sentence)) < 2:
                    # ignore sentences with only 1 real word
                    continue
                else:
                    s += 1

                # replace special patters
                text = dates.sub(u" _date_ ", sentence.text)
                text = times.sub(u" _time_ ", text)
                text = percents.sub(u" _percent_ ", text)
                text = years.sub(u" _year_ ", text)
                text = dollars.sub(u" _dollar_ ", text)
                text = emails.sub(u" _email_ ", text)
                text = urls.sub(u" _url_ ", text)
                    
                # remove last punctuation from text
                cleaned_text = re.sub(r"\s+", " ", punct.sub(u" ", text)).strip().lower()
                o.write("\t".join(fields[0:-1]) + "\t" +  str(s) + "\t" + cleaned_text + "\n")

    print("Finished parsing.")

    
if __name__ == "__main__":
    main()
