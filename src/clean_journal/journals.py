from __future__ import division, print_function, absolute_import
import os
import logging
import argparse
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import unicodedata

from src.collect_journal_keys import KeyCollector

logger = logging.getLogger(__name__)


class Journals(object):
    """
    provides an iterator over journal entries
    """
    def __init__(self, sites_dir=None, keys_file=None, init_stream=True, clean_method="none", verbose=False):
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
            clean_method: Can be either 'topic', 'sentiment', 'survival', or 'none' to specify the
                       method of cleaning the journal text (either for topic modeling,
                       sentiment analysis, survival analysis, or no cleaning, respectively).
                       This only needs to be set if the cleaning is being initiated
                       from the JournalsManager class (i.e. for parallel processing)
                       otherwise, if you're just interacting with a Journals object in a
                       for loop you can call whatever cleaning method you want.
            verbose:   Flag for whether or not to print progress to the log.
        """
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.sites_dir = sites_dir
        self.clean_method = clean_method
        self.verbose = verbose

        if sites_dir is not None and keys_file is not None:
            self.init_keys_file(keys_file)
            if init_stream:
                self.stream = self.init_generator()
            else:
                self.stream = None
        else:
            logger.info("No sites_dir or key_file specified, you won't be able to stream over a journal corpus, but you can use the clean_journal() method.")

        # store variables for cleaning text here, so it's accessible to all worker nodes
        self.lemmatizer = WordNetLemmatizer()

        # trying to err on the side of not removing too many 'stopwords'
        base_stopwords = stopwords.words("english")
        not_stopwords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves', u'you', u'your', u'yours', u'yourself', u'yourselves', u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself', u'what', u'who', u'whom', u'if', u'until', u'against', u'through', u'during', u'before', u'after', u'above', u'below', u'up', u'down',  u'on', u'off', u'over', u'under', u'again', u'further', u'when', u'where', u'why', u'own']
        self.stopword_set = set([w for w in base_stopwords if w not in not_stopwords])

        self.split_dash1 = re.compile(r"([a-zA-Z])([\-/])([a-zA-Z])")
        self.split_dash2 = re.compile(r"([0-9])([\-/])([a-zA-Z])")
        self.split_dash3 = re.compile(r"([a-zA-Z])([\-/])([0-9])")
        self.punct = re.compile(r"[^a-zA-Z_ ]") # keeping _'s in so we can use them as a special identifier

        # search for contractions
        self.iam = re.compile(r"'m")
        self.willnot = re.compile(r"won't")
        self.cannot = re.compile(r"can't")
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
        self.youall = re.compile(r"y'all|ya'll")
        self.youare = re.compile(r"you're")
        self.would = re.compile(r"'d")
        self.has = re.compile(r"'s")
        self.nt = re.compile(r"n't")
        self.will = re.compile(r"'ll")
        self.have = re.compile(r"'ve")
        self.s_apostrophe = re.compile(r"s' ")

        # here are some other regexs to capture certain patters, but it might be more efficient
        # to run these on the whole text final (via grep from command line)
        self.html = re.compile(r"<[^>]*>")
        self.years = re.compile(r"\b(18|19|20)[0-9]{2}\b")
        self.punct = re.compile(r"[^a-zA-Z_\s]")
        self.dollars = re.compile(r"\$[1-9][0-9,\. ]+")
        self.percent = re.compile(r"[1-9][0-9,\. ]+\%")
        self.times = re.compile(r"(2[0-3]|[01]?[0-9]):([0-5]?[0-9])")
        # email addresses? dates? numbers?

        # remove special characters that aren't needed (ex. &quot; &amp;)
        self.special_chars = re.compile(r"&[a-z]+;")

    def init_keys_file(self, keys_file):
        # check is a keys file is given (and if it really exists), if not create one
        if keys_file is None:
            raise ValueError("Cannot initialize a keys_file named None.")
        elif not os.path.isfile(keys_file):
            logger.info("The key's file given doesn't exists. Creating keys for all directories in sites_dir")
            self.keys_file = keys_file
            kc = KeyCollector(input_dir=self.sites_dir, output_filename=self.keys_file)
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
        if self.keys_file is None:
            raise ValueError("Cannot initialize stream of journals with a valid self.keys_file")

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
        This is just a method to call the appropriate method for
        cleaning the text data.

        TODO: A better solution is to have clean_method be an argument to clean_journal()
             rather than a class variable. Need to figure out how to pass
             function arguments in parallel when using JournalsManager.
        """
        if self.clean_method == "topic":
            return self.clean_journal_for_topic_modeling(journal)
        elif self.clean_method == "sentiment":
            raise ValueError("This method of cleaning the text data has not yet been implemented")
        elif self.clean_method == "survival":
            return self.clean_journal_for_survival_analysis(journal)
        elif self.clean_method == "none":
            return journal
        else:
            raise ValueError("clean_method class variable must be either topic, sentiment, or none")

    def clean_journal_for_survival_analysis(self, journal):
        """
        Helper function to clean the journal for use in survival analysis
        """
        # Here's an example. Please change this to whatever works for you!

        # step 1: clean up the text a little bit
        # remove emojis / unicode characters
        journal.body = unicodedata.normalize('NFKD', journal.body.decode('utf-8')).encode('ascii', 'ignore')
        # remove html
        journal.body = self.html.sub(" ", journal.body)
        # remove extra white space (i.e. "\n\r\t\f\v ")
        journal.body = re.sub(r"\s+", " ", journal.body)
        # remove punctation and numbers
        journal.body = self.punct.sub(" ", journal.body)


        # step 2: create features form the text
        # get number of words
        word_count = len(journal.body.split())
        # save word count in journal object
        journal.features.append(word_count)

        char_count = len(journal.body)
        journal.features.append(char_count)

        return journal

    def clean_journal_for_sentiment_analysis(self, journal):
        """
        Helper function to clean the journal for use in sentiment analysis
        """
        # TODO
        return journal

    def clean_journal_for_topic_modeling(self, journal, as_ascii=True, rm_html=True, rm_whitespace=True, split_dash=True, homogenize_patterns=True, expand_contractions=True, rm_special_chars=True, rm_punct=True, tokenize=True, rm_stopwords=True, lemmatize=True):
        """
        Helper function that implements how to clean text of a journal
        for work text that will be used in topic modeling

        Args:
            journal: the journal object to be modified

            Additional args are most just to help in testing,
            they're not meant to be interacted with directly.

        Returns: the journal object with the body variable cleaned and tokenized
                 so that the text is represented as a list of words

        """
        # This could be used to convert everything to ascii
        # REMOVE THIS ONCE WE FIGURE OUT HOW TO WORK WITH EMOJIS
        if as_ascii:
            journal.body = unicodedata.normalize('NFKD', journal.body.decode('utf-8')).encode('ascii', 'ignore')

        # remove html
        if rm_html:
            journal.body = self.html.sub(" ", journal.body)

        # remove extra white space (i.e. "\n\r\t\f\v ")
        if rm_whitespace:
            journal.body = re.sub(r"\s+", " ", journal.body)

        # split hyphens and slashes where appropriate
        if split_dash:
            journal.body = self.split_dash1.sub(r"\1 \3", journal.body)
            journal.body = self.split_dash2.sub(r"\1 \3", journal.body)
            journal.body = self.split_dash3.sub(r"\1 \3", journal.body)

        # homogenize special patterns
        if homogenize_patterns:
            journal.body = self.years.sub(" _year_ ", journal.body)
            journal.body = self.dollars.sub(" _dollars_ ", journal.body)
            journal.body = self.percent.sub(" _percent_ ", journal.body)
            journal.body = self.times.sub(" _time_ ", journal.body)

        # save emojis
        # ??? don't know how, what emojis are out there?

        # expand contractions
        # TODO Should am/has/is/etc just be deleted to save time? (they're stopwords)
        if expand_contractions:
            journal.body = self.iam.sub(" am", journal.body)
            journal.body = self.willnot.sub("will not", journal.body)
            journal.body = self.cannot.sub("can not", journal.body)
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
            journal.body = self.youare.sub("you are", journal.body)
            journal.body = self.would.sub(" would", journal.body)
            journal.body = self.will.sub(" will", journal.body)
            journal.body = self.s_apostrophe.sub("s has ", journal.body)
            journal.body = self.has.sub(" has", journal.body)
            journal.body = self.nt.sub(" not", journal.body)
            journal.body = self.have.sub(" have", journal.body)

        # remove special characters
        if rm_special_chars:
            journal.body = self.special_chars.sub(" ", journal.body)

        # remove remaining punctuation and numbers (except underscores)
        if rm_punct:
            journal.body = self.punct.sub(" ", journal.body)

        # tokenize
        if tokenize:
            journal.body = journal.body.split()
            journal.tokenized = True # setting this flag helps printing journal objects

        # remove stopwords
        if rm_stopwords:
            journal.body = [w for w in journal.body if w not in self.stopword_set]

        # lemmatize the remaining tokens
        # convert to lowercase, would like to do this earlier, but might get better lemmatizing results
        # if case is saved during lemmatization
        if lemmatize:
            journal.body = [self.lemmatizer.lemmatize(w, pos=self.get_wordnet_pos(p)).lower() \
                if self.get_wordnet_pos(p) != '' \
                else self.lemmatizer.lemmatize(w).lower() \
                for w, p in pos_tag(journal.body)]

        return journal

    def process_journals(self):
        """
        Clean and tokenize all journals in the stream.

        This method is solely needed for when parallel processing
        of journals is invoked from the JournalsManager class. Parallel
        nodes must process and store the results in RAM, then return them.

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
        self.features = []
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

    kf = "/home/srivbane/shared/caringbridge/data/dev/clean_journals/all_keys.tsv"
    j = Journals(sites_dir=args.sites_dir, keys_file=kf, verbose=args.verbose)
    for i, journal in enumerate(j.stream):
        if i > 3:
            break
        print(journal)




if __name__ == "__main__":
    main()
