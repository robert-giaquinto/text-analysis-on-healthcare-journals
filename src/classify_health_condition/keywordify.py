from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re

input_file = "parsed_site_data.txt"
learn_file = "cond_keywords.txt"
out_file = "custom_word_counts.txt"
keywords = ["cancer", "surgery", "injury", "breast", "stroke", "brain",
"transplantation", "leukemia", "lung", "lymphoma", "heart", "pancreatic",
"ovarian", "bone", "kidney", "myeloma", "skin", "bladder", "esophageal"]
punct_re = re.compile(r"[^a-zA-Z]")
stop_words = stopwords.words("english")
custom_stop_words = ["custom", "other", "condition"] + list('abcdefghijklmnopqrstuvwxyz')
stop_words_set = set([w for w in stop_words + custom_stop_words])
lemmatizer = WordNetLemmatizer()
word_counts = {}
with open(input_file, 'r') as fin, open(learn_file, 'w') as fout:
    for line in fin:
        out = line.split("\t")[0] + '\t' + line.split("\t")[1] + '\t'
        line = punct_re.sub(" ", line).lower()
        words = line.split()[1:]
        words = [w for w in words if w.lower() not in stop_words_set]
        # Assume everything is a noun
        words = [lemmatizer.lemmatize(w) for w in words]
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1

        for keyword in keywords:
            if not keyword in words:
                out += "0" + '\t'
            else:
                out += "1" + '\t'
        out += '\n'
        fout.write(out)
        

with open(out_file, 'w') as fout:
    for word, count in word_counts.iteritems():
        out = word + '\t' + str(count) + '\n'
        fout.write(out)



