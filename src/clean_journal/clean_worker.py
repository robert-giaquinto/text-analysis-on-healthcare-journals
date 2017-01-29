from __future__ import division, print_function, absolute_import
import os
import logging
import argparse
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re
import unicodedata

from src.journal import Journal
from src.clean_journal.first_names import FirstNames

logger = logging.getLogger(__name__)


class JournalCleaningWorker(object):
    """
    provides an iterator over journal entries
    """
    def __init__(self, input_file=None, output_file=None, clean_method="none", init_stream=True, verbose=False):
        """
        Args:
            input_file:   File of journals that have been parsed from json, assumes
                          file is tab seperated with the columns:

                          siteId    userID    journalId    createdAt    text

            output_file:  File of where to save the cleaned journals. Will contain
                          the same keys as the input file. If clean method is surival
                          then the file won't contain text, instead it will have the features
                          derived in the cleaning method for survival analysis. Otherwise,
                          the output file will contain the cleaned up text.

            clean_method: Can be either 'topic', 'sentiment', 'survival', or 'none' to specify the
                       method of cleaning the journal text (either for topic modeling,
                       sentiment analysis, survival analysis, or no cleaning, respectively).
                       This only needs to be set if the cleaning is being initiated
                       from the JournalsManager class (i.e. for parallel processing)
                       otherwise, if you're just interacting with a Journals object in a
                       for loop you can call whatever cleaning method you want.
            init_stream: True if you plan to clean the journals in input_file. You can set to
                         False if you plan to just use this class for the clean_journal methods
                         and don't plan to pass a whole input file to process.
            verbose:   Flag for whether or not to print progress to the log.
        """
        if verbose:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.input_file = input_file
        self.output_file = output_file
        self.clean_method = clean_method
        self.verbose = verbose

        if input_file is not None and init_stream:
            self.stream = self.init_generator()
        else:
            self.stream = None

        if clean_method != "none":
            self.init_clean_journals()

    def init_clean_journals(self):
        # store variables for cleaning text here, so it's accessible to all worker nodes
        self.lemmatizer = WordNetLemmatizer()

        # trying to err on the side of not removing too many 'stopwords'
        base_stopwords = stopwords.words("english")
        custom_stopwords = ['got', 'get', 'til', 'also', 'would', 'could', 'should', 'really',
                            'didnt', 'cant', 'thats', 'doesnt', 'didnt', 'wont', 'wasnt', 'hows',
                            'hadnt', 'hasnt', 'willnt', 'isnt', 'arent', 'werent', 'havent',
                            'wouldnt', 'couldnt', 'shouldnt',  'shouldve', 'couldve', 'wouldve', 
                            'theres', 'whats', 'whens', 'whos', 'wheres'] + list('abcdefghjklmnopqrstuvwxyz')
        self.first = FirstNames()
        not_stopwords = [u'i', u'me', u'my', u'myself', u'we', u'our', u'ours', u'ourselves',
                         u'they', u'them', u'themselves',
                         u'you', u'your', u'yours', u'yourself', u'yourselves',
                         u'he', u'him', u'his', u'himself', u'she', u'her', u'hers', u'herself',
                         u'against', u'through', u'during', u'before', u'after', u'above', u'below',
                         u'up', u'down', u'over', u'under', u'again', u'why']
        self.stopword_set = set([w for w in base_stopwords + custom_stopwords if w not in not_stopwords])

        self.split_dash1 = re.compile(r"([a-z])([\-/])([a-z])", re.IGNORECASE)
        self.split_dash2 = re.compile(r"([0-9])([\-/])([a-z])", re.IGNORECASE)
        self.split_dash3 = re.compile(r"([a-z])([\-/])([0-9])", re.IGNORECASE)
        self.weekend = re.compile(r"week\-end[a-z]*[ ,\.]", re.IGNORECASE)
        self.xray = re.compile(r"x-ray[a-z]*[ ,\.]|xray[a-z]*[ ,\.]|x ray[a-z]*[ ,\.]", re.IGNORECASE)
        self.email = re.compile(r"e-mail[a-z][ ,\.]* |email[a-z]*[ ,\.]", re.IGNORECASE)
        
        self.punct = re.compile(r"[^a-zA-Z_ ]") # keeping _'s in so we can use them as a special identifier

        # search for contractions
        self.im = re.compile(r"im", re.IGNORECASE)
        self.iam = re.compile(r"'m", re.IGNORECASE)
        self.ill = re.compile(r"ill", re.IGNORECASE)
        self.youll = re.compile(r"youll", re.IGNORECASE)
        self.ive = re.compile(r"ive", re.IGNORECASE)
        self.hes = re.compile(r"hes", re.IGNORECASE)
        self.shes = re.compile(r"shes", re.IGNORECASE)
        self.weve = re.compile(r"weve", re.IGNORECASE)
        self.youve = re.compile(r"youve", re.IGNORECASE)
        self.willnot = re.compile(r"won't", re.IGNORECASE)
        self.cannot = re.compile(r"can't", re.IGNORECASE)
        self.itis = re.compile(r"it's", re.IGNORECASE)
        self.letus = re.compile(r"let's", re.IGNORECASE)
        self.heis = re.compile(r"he's", re.IGNORECASE)
        self.howis = re.compile(r"how's", re.IGNORECASE)
        self.thatis = re.compile(r"that's", re.IGNORECASE)
        self.thereis = re.compile(r"there's", re.IGNORECASE)
        self.whatis = re.compile(r"what's", re.IGNORECASE)
        self.whereis = re.compile(r"where's", re.IGNORECASE)
        self.whenis = re.compile(r"when's", re.IGNORECASE)
        self.whois = re.compile(r"who's", re.IGNORECASE)
        self.whyis = re.compile(r"why's", re.IGNORECASE)
        self.youall = re.compile(r"y'all|ya'll", re.IGNORECASE)
        self.youare = re.compile(r"you're", re.IGNORECASE)
        self.would = re.compile(r"'d", re.IGNORECASE)
        self.has = re.compile(r"'s", re.IGNORECASE)
        self.nt = re.compile(r"n't", re.IGNORECASE)
        self.will = re.compile(r"'ll", re.IGNORECASE)
        self.have = re.compile(r"'ve", re.IGNORECASE)
        self.s_apostrophe = re.compile(r"s' ", re.IGNORECASE)

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
        if self.input_file is None:
            raise ValueError("Cannot intialize stream of journals without a valid file.")

        with open(self.input_file, 'r') as fin:
            for line in fin:
                siteId, userId, journalId, createdAt, body = line.replace('\n', '').strip().split('\t')
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
            # return clean_journal_for_sentiment_analysis(journal)
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

    def clean_journal_for_topic_modeling(self, journal, as_ascii=True, rm_html=True, rm_whitespace=True, split_dash=True, homogenize_patterns=True, expand_contractions=True, rm_special_chars=True, rm_punct=True, tokenize=True, rm_stopwords=True, rm_names=True, lemmatize=True):
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
            journal.body = self.weekend.sub("weekend ", journal.body)
            journal.body = self.xray.sub("x_ray ", journal.body)
            journal.body = self.email.sub("email ", journal.body)
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
            journal.body = self.im.sub("i", journal.body)
            journal.body = self.ill.sub("i", journal.body)
            journal.body = self.youll.sub("you", journal.body)
            journal.body = self.ive.sub("i", journal.body)
            journal.body = self.hes.sub("he", journal.body)
            journal.body = self.shes.sub("she", journal.body)
            journal.body = self.weve.sub("we", journal.body)
            journal.body = self.youve.sub("you", journal.body)
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
            journal.body = [w for w in journal.body if w.lower() not in self.stopword_set]

        # remove first names
        if rm_names:
            journal.body = ["_name_" if w.lower() in self.first.names else w for w in journal.body]

        # lemmatize the remaining tokens
        # convert to lowercase, would like to do this earlier, but might get better lemmatizing results
        # if case is saved during lemmatization
        if lemmatize:
            journal.body = [self.lemmatizer.lemmatize(w, pos=self.get_wordnet_pos(p)).lower() \
                if self.get_wordnet_pos(p) != '' \
                else self.lemmatizer.lemmatize(w).lower() \
                for w, p in pos_tag(journal.body)]

        return journal

    def clean_and_save(self):
        """
        Run the cleaning method on all journals in input_file.

        Args:
            None
        Returns: A list of lists. Each elt in list is a journal, the sublists
                 are the tokenized terms from the journal's text.
        """
        # check to make sure the stream of journals exists
        if self.stream is None:
            self.stream = self.init_generator()

        if self.output_file is None:
            raise ValueError("Cannot process the journals unless you specify a file to save the results in.")

        logger.info("Outputting: " + self.output_file)
        with open(self.output_file, 'wb') as fout:
            for journal in self.stream:
                journal = self.clean_journal(journal)
                output = journal.siteId + '\t' + journal.userId + '\t' + journal.journalId + '\t' + journal.createdAt + '\t'

                if self.clean_method == "survival":
                    # for survival analysis, don't save the text, just tab separate the features
                    output += '\t'.join(journal.features)
                else:
                    # for other tasks (like topic modeling) we want the journal body saved
                    if len(journal.body) == 0:
                        continue

                    # else save the output to the file
                    output += ' '.join(journal.body)

                fout.write(output + '\n')


def main():
    parser = argparse.ArgumentParser(description='Each worker contains the methods used to clean the journal data for analysis. Workers can be called to work in parallel via clean_manager.py.')
    parser.add_argument('-i', '--input_file', type=str, help='File with a journal on each line.')
    parser.add_argument('-o', '--output_file', type=str, help='Where to save the cleaned results.')
    parser.add_argument('-c', '--clean_method', type=str, default='topic', help='Method of cleaning each journal. Default is the topic modeling cleaning method "topic".')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    print('clean_worker.py')
    print(args)

    j = JournalCleaningWorker(input_file=args.input_file, output_file=args.output_file, clean_method=args.clean_method, init_stream=True, verbose=args.verbose)
    
    for i, journal in enumerate(j.stream):
        if i > 3:
            break
        print("Original journal:\n")
        print(journal)
        print("\nCleaned journal:\n")
        print(j.clean_journal(journal))


if __name__ == "__main__":
    main()
