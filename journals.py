from __future__ import division, print_function
import os
import logging
import argparse
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re

from parse_journal.collect_journal_keys import KeyCollector

logger = logging.getLogger(__name__)


# TODO should probabily check that the journal files don't contain carriage returns

class Journals(object):
    """
    provides an iterator over journal entries
    """
    def __init__(self, sites_dir, keys_file=None, init_stream=True, verbose=False):
        """
        Args:
            sites_dir:   Folder where all of the folders for each site exist
            keys_file:   File containing a list of the keys for all the journal files
                         you want to iterate over.
                         If no argument given, a keys file will be created for all sites
                         in the sites_dir.
            init_stream: True/False do you want to initialize the stream of journals.
                         Use init_stream=True (default) when you just want to interact
                         with the journal entries.
                         Use init_stream=False if you're running Journals in parallel.
                         To run in parallel  the initialization needs to occur later
                         (and not when the Journals instance is being declared).
            verbose:   Flag for whether or not to print progress to the log.
        """
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.sites_dir = sites_dir
        self.verbose = verbose
        self.init_keys_file(keys_file)

        if init_stream:
            self.stream = self.init_generator()
        else:
            self.stream = None

        # store variables for cleaning text here, so it's accessible to all worker nodes
        self.lemmatizer = WordNetLemmatizer()        
        self.stopword_set = set(stopwords.words("english"))
        self.split_dash = re.compile(r"([a-zA-Z])([-/])([a-zA-Z])")
        self.punct = re.compile(r"[^a-zA-Z_ ]") # keeping _'s in so we can use them as a special identifier

        self.iam = re.compile(r"'m")
        self.willnot = re.compile(r"wont't")
        self.nt = re.compile(r"n't")
        self.itis = re.compile(r"it's")
        self.letus = re.compile(r"let's")
        self.heis = re.compile(r"he's")
        self.howis = re.compile(r"how's")
        self.thatis = re.compile(r"that's")
        self.thereis = re.compile(r"there's")
        self.whatis = re.compile(r"what's")
        self.whereis = re.compile(r"where's")
        self.whenis = re.compile(r"when's")
        self.whois = re.compile(r"who's")
        self.whyis = re.compile(r"why's")
        self.youall = re.compile(r"y'all")
        self.would = re.compile(r"'d")
        self.has = re.compile(r"'s")
        self.youare = re.compile(r"you're")
        self.will = re.compile("'ll")
        self.s_apostrophe = re.compile("s' ")

        # here are some other regexs to capture certain patters, but it might be more efficient
        # to run these on the whole text final (via grep from command line)
        self.html = re.compile(r"<[^>]*>")
        self.years = re.compile(r"\b(18|19|20)[0-9]{2}\b")
        self.punct = re.compile(r"[^a-zA-Z_\s]")
        #self.dollars = re.compile(r"\$[1-9][0-9,\.]+")
        # email addresses? times? dates? numbers? 

    def init_keys_file(self, keys_file):
        # check is a keys file is given (and if it really exists), if not create one
        if keys_file is None:
            logger.info("No key's file given. Creating keys for all directories in sites_dir")
            self.keys_file = "/home/srivbane/shared/caringbridge/data/clean_journals/all_keys.tsv"
            kc = KeyCollector(input_dir=self.sites_dir, output_filename=self.keys_file, verbose=self.verbose)
            kc.collect_keys()
        elif not os.path.isfile(keys_file):
            logger.info("The key's file given doesn't exists. Creating keys for all directories in sites_dir")
            self.keys_file = keys_file
            kc = KeyCollector(input_dir=self.sites_dir, output_filename=self.keys_file, verbose=self.verbose)
            kc.collect_keys()
        else:
            self.keys_file = keys_file
            logger.info("Keys file found, using the file you specified.")
        
    def init_generator(self):
        """
        Return a generator over the journal files
        
        ex:
        journals = Journals(sites_dir = '/home/srivbane/shared/caringbridge/data/parsed_json')
        for journal in journals.stream:
            print(journal.siteId)
            print(journal.journalId)
            print(journal.body)
        """
        with open(self.keys_file, 'r') as key_file:
            for line in key_file:
                # pull out the keys to a journal and unpack the values into variables 
                journal_keys = line.replace('\n', '').strip().split('\t')
                siteId, userId, journalId, createdAt = journal_keys
                
                filename = os.path.join(self.sites_dir, siteId, '_'.join(journal_keys))
                with open(filename, 'r') as journal_file:
                    # body of journal may be spread out of multiple lines, remove all blank lines
                    body = filter(None, [j.replace('\n', ' ').strip() for j in journal_file.readlines()])
                    # paste everything back together
                    body = '\n'.join(body)
                    
                journal = Journal(siteId=siteId, userId=userId, journalId=journalId, createdAt=createdAt, body=body)
                yield journal

    def get_wordnet_pos(self, treebank_tag):
        """
        Need to know the part of speech of each word to properly lemmatize.
        this function standardizes the POS codes so that they're understandable
        by the lemmatizing function.
        :param treebank_tag:
        :return:
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return ''

    def clean_journal(self, journal):
        """
        Implement function for how to clean text of a journal

        Args:
            journal: the journal object to be modified
        Returns: the journal object with the body variable cleaned and tokenized
                 so that the text is represented as a list of words
        
        """
        # split hyphens and slashes
        journal.body = self.split_dash.sub("$1 $3", journal.body)
        
        # homogenize special patterns
        journal.body = self.years.sub(" _year_ ", journal.body)

        # save emojis
        # ??? don't know how, what emojis are out there?

        # expand contractions
        journal.body = self.iam.sub(" am", journal.body)
        journal.body = self.willnot.sub("will not", journal.body)
        journal.body = self.nt.sub(" not", journal.body)
        journal.body = self.itis.sub("it is", journal.body)
        journal.body = self.letus.sub("let us", journal.body)
        journal.body = self.heis.sub("he is", journal.body)
        journal.body = self.howis.sub("how is", journal.body)
        journal.body = self.thatis.sub("that is", journal.body)
        journal.body = self.thereis.sub("there is", journal.body)
        journal.body = self.whatis.sub("what is", journal.body)
        journal.body = self.whereis.sub("where is", journal.body)
        journal.body = self.whenis.sub("when is", journal.body)
        journal.body = self.whois.sub("who is", journal.body)
        journal.body = self.whyis.sub("why is", journal.body)
        journal.body = self.youall.sub("you all", journal.body)
        journal.body = self.would.sub(" would", journal.body)
        journal.body = self.youare.sub("you are", journal.body)
        journal.body = self.will.sub(" will", journal.body)
        journal.body = self.s_apostrophe.sub(" has", journal.body)
        journal.body = self.has.sub(" has", journal.body)

        # remove html
        journal.body = self.html.sub(" ", journal.body)

        # remove remaining punctuation (except underscores)
        journal.body = self.punct.sub(" ", journal.body)

        # to lowercase and tokenize
        journal.body = journal.body.lower().split()
        journal.tokenized = True # setting this flag helps printing journal objects

        # remove stopwords
        journal.body = [w for w in journal.body in w not in self.stopword_set]
        # lemmatize the remaining tokens
        journal.body = [self.lemmatizer.lemmatize(w, pos=self.get_word_net_pos(p) if self.get_wordnet_pos(p) != '' \
                                                      else self.lemmatizer.lemmatize(w) \
                                                      for w, p in pos_tag(journal.body)]
        return journal

    def process_journals(self):
        """
        Clean and tokenize all journals in the stream.
        Args:
            None
        Returns: A list of lists. Each elt in list is a journal, the sublists 
                 are the tokenized terms from the journal's text.
        """
        # check to make sure the stream of journals exists
        if self.stream is None:
            self.stream = self.init_generator()

        rval = []
        for journal in self.stream:
            rval.append(self.clean_journal(journal))
        return rval

    
class Journal(object):
    """
    Just a simple structure to hold journal information
    """
    def __init__(self, siteId=None, userId=None, journalId=None, createdAt=None, body=None):
        self.siteId = siteId
        self.userId = userId
        self.journalId = journalId
        self.createdAt = createdAt
        self.body = body
        self.tokenized = False

    def __repr__(self):
        return "Journal Object"

    def __str__(self):
        return "\nsiteId: " + str(self.siteId) +\
            "\nuserId: " + str(self.userId) +\
            "\njournalId: " + str(self.journalId) +\
            "\ncreatedAt: " + str(self.createdAt) +\
            "\nbody: " + ' '.join(self.body) if self.tokenized else self.body
        
def main():
    parser = argparse.ArgumentParser(description='An example of using the iterator over journal files.')
    parser.add_argument('-i', '--sites_dir', type=str, help='Path to directory containing all the site directories.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()
    
    print('journals.py')
    print(args)

    j = Journals(sites_dir=args.sites_dir)
    for i, journal in enumerate(j.stream):
        if i > 3:
            break
        print(journal)




if __name__ == "__main__":
    main()
