import nltk
dl = ["cess_esp", "ptb", "reuters", "sentiwordnet", "stopwords", "treebank", "universal_treebanks_v2", "verbnet", "wordnet", "words", "hmm_treebank_pos_tagger", "maxent_treebank_pos_tagger", "snowball_data", "universal_tagset"]
for d in dl:
    nltk.download(d)
