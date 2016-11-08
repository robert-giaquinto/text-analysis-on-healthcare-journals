from __future__ import division, print_function, absolute_import
import argparse
import os
from math import ceil, log
from gensim import models, utils
import numpy as np
import logging

from src.topic_model.documents import Documents
from src.utilities import pickle_it


# initialize logging
logger = logging.getLogger(__name__)

gensim_logger1 = logging.getLogger('gensim.models.ldamodel')
gensim_logger2 = logging.getLogger('gensim.models.ldamulticore')

# filter out the un-needed  gensim logging messages
class NoGensimFilter(logging.Filter):
    def filter(self, record):
        #useless = record.getMessage().startswith('PROGRESS')  or record.funcName == "blend" or record.funcName == "show_topics"
        useless = record.funcName == "blend" or record.funcName == "show_topics"
        return not useless

gensim_logger1.addFilter(NoGensimFilter())
gensim_logger2.addFilter(NoGensimFilter())



class GensimLDA(object):
    """
    This is a wrapper around Gensim's LDA model.
    Provides some new features, and allows for training until convergence
    """
    def __init__(self, docs, n_workers=1, verbose=False):
        """
        Args:
            docs: a Documents object
            n_workers: How many cores to use in training the LDA model.
            verbose: Should progress be logged?
        """
        if verbose:
            logging.basicConfig(format='%(name)s : %(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.n_workers = n_workers
        self.docs = self._init_docs(docs)

        self.num_train = docs.num_train
        self.num_test = docs.num_test
        self.model = None
        self.topic_terms = None
        self.topic_term_method = "weighted" # weighted is a tf-idf weighting of terms for each topic, as opposed to standard probability
        self.num_topics = None

        # dictionarys to hold word cooccurences (speeds up NPMI calculation)
        self.doc_token2freq, self.token2freq = None, None

    def _init_docs(self, docs):
        """
        Just check to make sure docs.vocab, docs.train_bow, and docs.test_bow are loaded
        """
        if docs.vocab is None:
            docs.load_vocab()

        if docs.train_bow is None or docs.test_bow is None:
            docs.load_bow()

        if docs.train_keys is None or docs.test_keys is None:
            docs.load_keys()

        return docs

    def fit_partial(self, num_topics, perplexity_threshold, evals_per_pass, chunksize, max_passes=3):
        """
        Fit a single LDA model given some parameters.

        Args:
            num_topics: How many topics should the LDA model find.
            perplexity_threshold: Training will be considered converged once changes in
                                  perplexity are less than perplexity_threshold.
            evals_per_pass: How many times to evaluate perplexity per pass through the corpus?
            chunksize: Number of documents to fit at a time (mini-batch size)
        Returns: a dictionary containing a few performance statistics monitoring
                 how well + quickly the model converged
        """
        convergence = {'perplexities': [], 'docs_seen': []}

        logger.info("Initializing model.")
        if self.n_workers == 1:
            model = models.ldamodel.LdaModel(id2word=self.docs.vocab,
                                             num_topics=num_topics,
                                             chunksize=chunksize,
                                             eval_every=None)
        else:
            model = models.ldamulticore.LdaMulticore(id2word=self.docs.vocab,
                                                     num_topics=num_topics,
                                                     workers=self.n_workers,
                                                     chunksize=chunksize,
                                                     eval_every=None)

        if perplexity_threshold == 0.0:
            # don't stop early, just do 1 pass all in gensim
            logger.info("Running for just one full pass through data, not stopping to check for perplexity convergence")
            model.eval_every = evals_per_pass
            model.update(corpus=self.docs.train_bow)
            convergence['perplexities'] = [self._perplexity_score(model, self.docs.test_bow, total_docs=self.num_test + self.num_train)]
            convergence['docs_seen'] = self.num_train
            return model, convergence

        # ELSE: run for a chunk of data, check for convergence
        logger.info("Running for at least one pass, checking for perplexity convergence with threshold of " + str(perplexity_threshold))
        converged = False
        prev_perplexity, perplexity = None, None
        docs_seen = 0
        for p in range(max_passes):
            logger.info("Pass: " + str(p))

            # train on num_train / evals_per pass documents at the time, then evaluate perplexity
            chunk_stream = utils.grouper(self.docs.train_bow, int(ceil(self.num_train / evals_per_pass)), as_numpy=False)
            for i, chunk in enumerate(chunk_stream):
                model.update(corpus=chunk, chunks_as_numpy=False)
                docs_seen += len(chunk)

                # measure perplexity after training of the most recent chunk
                perplexity = self._perplexity_score(model, self.docs.test_bow, self.num_test + self.num_train)
                logger.info("Pass: %d, chunk: %d/%d, perplexity: %f", p, i+1, evals_per_pass, round(perplexity,2))
                convergence['perplexities'].append(perplexity)
                convergence['docs_seen'].append(docs_seen)

                # check for convergence
                converged = prev_perplexity and abs(perplexity - prev_perplexity) < perplexity_threshold
                if (p > 0 or i+1 == evals_per_pass) and converged:
                    logger.info("Converged after " + str(p) + "passes, and seeing " + str(docs_seen) + " documents.")
                    logger.info("Convergence perplexity: " + str(perplexity))

                prev_perplexity = perplexity

        return model, convergence

    def fit(self, num_topics, chunksizes=None, perplexity_threshold=None, evals_per_pass=4):
        """
        Find the best LDA model for a range of num_topics and chunksizes
        """
        if isinstance(num_topics, int):
            num_topics = [num_topics]
        if not isinstance(num_topics, list):
            raise ValueError("num_topics must either be a list of ints or a single int")

        if isinstance(chunksizes, int):
            chunksizes = [chunksizes]
        if not isinstance(chunksizes, list):
            raise ValueError("chunksizes must be either a list of ints or a single int")

        model = None
        best_perplexity = float("inf")
        performance = []
        for n in num_topics:
            for c in chunksizes:
                logger.info("Training LDA model with " + str(n) + " topics, and a mini-batch size of " + str(c) + ".")
                model, convergence = self.fit_partial(num_topics=n,
                                                      perplexity_threshold=perplexity_threshold,
                                                      evals_per_pass=evals_per_pass,
                                                      chunksize=c)
                performance.append({'num_topics': n,
                                    'chunksize': c,
                                    'perplexities': convergence['perplexities'],
                                    'docs_seen': convergence['docs_seen']})
                perplexity = convergence['perplexities'][-1]
                if perplexity < best_perplexity:
                    logger.info("New best model found")
                    self.model = model
                    best_perplexity = perplexity

        # set topic terms
        if self.topic_term_method == "weighted":
            self.topic_terms = self._weighted_topic_terms()
        else:
            self.topic_terms = self._unweighted_topic_terms()
        # save best number of topics found
        self.num_topics = self.model.num_topics

        return performance

    def _perplexity_score(self, model, X, total_docs=None):
        """
        Calculate perplexity on a set of documents X
        """
        if total_docs is None:
            total_docs = sum(1 for _ in X)

        corpus_words = sum(ct for doc in X for _, ct in doc)
        subsample_ratio = 1.0 * total_docs / len(X)
        perword_bound = model.bound(X, subsample_ratio=subsample_ratio) / (subsample_ratio * corpus_words)
        perplexity = np.exp2(-perword_bound)
        return perplexity

    def topic_scores(self, performance_metric="NPMI"):
        """
        Calls appropriate scoring method based on argument performance_metric
        Args:
            performance_metric:

        Returns: a number a list of how each topic performed
        """
        if performance_metric == "NPMI":
            return self._npmi_score()
        elif performance_metric == "W2V":
            raise ValueError("word2vec evaluation of topics not implemented yet")
            #return self._w2v_score()
        else:
            raise ValueError("performance metric for scoring a set of topic terms must be either NPMI or W2V")

    def _npmi_score(self):
        """
        Normalized pair-wise mutual information method of scoring
        the quality of topics.
        Note: this is not being calculated in the traditional way. This doesn't
              use context windows for calculating word co-occurences, rather
              words count as co-occuring if this both appear in the document, the
              denominator in this probability is how many documents there are.
        """
        if self.model is None:
            raise ValueError("You must call fit before you can score the quality of topics.")

        if self.doc_token2freq is None or self.token2freq is None:
            self.token2freq, self.doc_token2freq = get_word_counts(self.docs.train_bow, self.docs.vocab)

        term_count = sum([ct for ct in self.token2freq.itervalues()])

        epsilon = 0.00001
        npmis = []
        for topic in self.topic_terms:
            summation_terms = []
            for j, term_j in enumerate(topic[1:10]):
                for term_i in topic[0:j]:
                    # lookup individual word probabilities
                    pr_j = 1.0 * self.token2freq[term_j] / term_count
                    pr_i = 1.0 * self.token2freq[term_i] / term_count

                    docs_with_cooccur = 0
                    cooccur_ct = 0
                    for doc in self.doc_token2freq:
                        if term_j in doc and term_i in doc:
                            docs_with_cooccur += 1
                    pr_j_and_i = 1.0 * docs_with_cooccur / len(self.doc_token2freq)
                    numerator = log((pr_j_and_i + epsilon) / (pr_i * pr_j), 2)
                    denominator = -1.0 * log(pr_j_and_i + epsilon, 2)
                    summation_terms.append(numerator / denominator)
            npmi = (1.0 / 45.0) * sum(summation_terms)
            npmis.append(npmi)
        return npmis

    def save_topic_terms(self, output_filename, metric="NPMI"):
        """
        Save the top 10 words for each topic.
        """
        if self.topic_terms is None:
            raise ValueError("Topic terms not defined, call fit first")

        # put the topic in order of highest to lowest based on a metric
        if metric != "none":
            scores = self.topic_scores(metric)
            sorted_term_scores = sorted(zip(self.topic_terms, scores), key=lambda x: x[1], reverse=True)
            sorted_terms, sorted_scores  = zip(*sorted_term_scores)
            sorted_ids = [rank for rank, _ in sorted(enumerate(scores), key = lambda x: x[1], reverse=True)]
            # write to file, order topics by ranking
            with open(output_filename, "wb") as f:
                f.write("topic_id,topic_rank,score," + ','.join(['term' + str(i) for i in range(10)]) + "\n")
                topic_rank = 1
                for terms, topic_id, score in zip(sorted_terms, sorted_ids, sorted_scores):
                    f.write(str(topic_id) + "," + str(topic_rank) + "," + str(score) + "," + ','.join(terms) + "\n")
                    topic_rank += 1
        else:
            with open(output_filename, "wb") as f:
                f.write("topic_id," + ','.join(['term' + str(i) for i in range(10)]) + "\n")
                for topic_id, term_arr in enumerate(self.topic_terms):
                    f.write(str(topic_id) + "," + ','.join(term_arr) + "\n")

    def save_word_topic_probs(self, output_filename, metric="NPMI"):
        """
        Save the beta parameter (probability of each word belonging to each topic
        to a file.
        """
        betas = self.get_beta()
        vocab_words = [wd for key, wd in sorted(self.model.id2word.iteritems())]
        if metric != "none":
            topic_scores = self.topic_scores(metric)

        with open(output_filename, "wb") as f:
            if metric != "none":
                f.write("topic_weight," + ','.join([w for w in vocab_words]) + "\n")
                for topic_score, word_probs in zip(topic_scores, betas):
                    f.write(str(topic_score) + "," + ','.join([str(p) for p in word_probs]) + "\n")
            else:
                f.write(','.join([w for w in vocab_words]) + "\n")
                for word_probs in betas:
                    f.write(','.join([str(p) for p in word_probs]) + "\n")

                    
                    
    def save_doc_topic_probs(self, bow_generator, keys_generator, output_filename):
        """
        Save the probability distribution of topics over each of the documents
        Args:
            bow_generator: a bag-of-words generator. User either:
                           from gensim import corpora
                           bow_generator = corpora.MmCorpus("my_bag_of_words.mm")
                           or:
                           docs = Documents(...)
                           bow_generator = docs.train_bow # or bow = docs.test_bow
            keys_generator: a generator that return one list of keys to each document
                            at a time. Best to use:
                            docs = Documents(...)
                            keys_generator = docs.train_keys
                            or:
                            keys_generator = docs.test_keys
            output_filename: name of file to write results to.
        Returns: Nothing, results written to file
        """
        with open(output_filename, "wb") as fout:
            chunk_stream = utils.grouper(bow_generator, chunksize=25000, as_numpy=False)
            for chunk in chunk_stream:
                topic_dist = self.get_theta(list(chunk))
                for doc_keys, td in zip(keys_generator, topic_dist):
                    fout.write(','.join(doc_keys) + "," + ','.join([str(prob) for prob in td]) + "\n")

    def get_beta(self):
        """
        Get matrix with distribution of words over topics.
        """
        betas = []
        for topic_id in range(self.model.num_topics):
            topic = self.model.state.get_lambda()[topic_id]
            topic = topic / topic.sum()  # normalize to probability dist
            betas.append(topic)
        return betas

    def get_theta(self, bows):
        """
        Theta is the parameter for topic distribution over each document
        This can be a little faster than gensim's native 1 document at a time approach to inference
        :param bows: should be a list of bag of words vectors loaded into memory
        :return:
        """
        gamma, _ = self.model.inference(bows, collect_sstats=False)
        # normalize
        theta = []
        for g in gamma:
            topic_dist = g / sum(g)
            theta.append(topic_dist)
        if len(theta) == 1:
            return theta[0]
        else:
            return theta

    def _weighted_topic_terms(self):
        """
        Return top 10 topic terms from model using the term-score weighted
        (Blei and Lafferty 2009, equation 3)

        This is equivalent to using TF-IDF to determine the top terms

        Args: None
        Returns: list topics where each is a list of 10 terms
        """
        if self.model is None:
            raise ValueError("Fit must be called to set the topics.")

        # first, pull out probabilities for all words and topics
        vocab = self.model.id2word
        betas = self.get_beta()
        vocab_words = np.array([wd for key, wd in sorted(vocab.iteritems())])
        term_scores = self._term_scores(betas)

        # select top 10 terms with highest term-score in each topic
        top_terms = []
        for topic in term_scores:
            top_ten_indices = np.argpartition(topic, -10)[-10:][::-1]
            top_ten_terms = vocab_words[top_ten_indices].tolist()
            top_terms.append(top_ten_terms)
        return top_terms

    def _term_scores(self, betas):
        """
        TF-IDF type calculation for determining top topic terms
        from Blei and Lafferty 2009, equation 3

        Args:
            betas:

        Returns:
        """
        denom = np.power(np.prod(betas, axis=0), 1.0 / len(betas))
        if np.any(denom == 0):
            denom +=  0.0000001
        term2 = np.log(np.divide(betas, denom))
        return np.multiply(betas, term2)

    def _unweighted_topic_terms(self):
        """
        Just use the standard approach of selecting words with highest probability

        Returns: list topics where each is a list of 10 terms
        """
        if self.model is None:
            raise ValueError("Fit must be called to set the topics.")
        # just extract the topics from based on probability score
        raw_topics = self.model.show_topics(num_topics=-1, num_words=10, formatted=False)
        rval = [[word for word, _ in topic] for _, topic in raw_topics]
        return rval


def get_word_counts(x, vocab):
    # create word frequency lookups for faster computing of NPMI scores
    token2freq = {}
    doc_token2freq = []
    term_count = 0  # number of tokens in x
    for doc in x:
        term_freq_pairs = []
        for term_id, f in doc:
            freq = int(f)
            term_count += freq
            term = vocab[term_id]
            term_freq_pairs.append((term, freq))
            if term not in token2freq:
                token2freq[term] = freq
            else:
                token2freq[term] += freq
        doc_token2freq.append(dict(term_freq_pairs))
    return token2freq, doc_token2freq


def main():
    parser = argparse.ArgumentParser(description='Wrapper around the gensim LDA model.')
    parser.add_argument('-j', '--journal_file', type=str, help='Full path to the journal file to extract tokens from.')
    parser.add_argument('-d', '--data_dir', type=str, help='Directory of where to save or load bag-of-words, vocabulary, and model performance files.')
    parser.add_argument('-k', '--keep_n', type=int, help='How many terms (max) to keep in the dictionary file.')
    parser.add_argument('--num_test', type=int, default=0, help="Number of documents to hold out for the test set.")
    parser.add_argument('--num_docs', type=int, help="Number of documents in the journal file (specifying this can speed things up).")
    parser.add_argument('--num_topics', type=int, nargs='+', default=[25], help="Number of topics to extract. Multiple arguments can be given to test a range of parameters.")
    parser.add_argument('--n_workers', type=int, default=1, help="Number of cores to run on.")
    parser.add_argument('--chunksizes', type=int, nargs='+', default=[1024], help="Mini-batch size for model training. Multiple arguments can be given to test a range of parameters.")
    parser.add_argument('--threshold', type=float, default=0.01, help="A difference in perplexity between iterations of this amount will signal convergence.")
    parser.add_argument('--evals_per_pass', type=int, default=4, help="How many times to check model perplexity per passes over the full dataset.")
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.add_argument('--rebuild', dest="rebuild", action="store_true", help='Add this flag to rebuild the bag-of-words and vocabulary, even if copies of the files already exists.')
    parser.add_argument('--no_shuffle', dest="shuffle", action="store_false", help='Add this flag to not shuffle the data before splitting it into training and test BOW files. This is not recommended, but can save a lot of time on big datasets.')
    parser.add_argument('--no_score', dest="score_topics", action="store_false", help='Add this flag to not score and rank the coherence of topics.')
    parser.set_defaults(verbose=False)
    parser.set_defaults(rebuild=False)
    parser.set_defaults(shuffle=True)
    parser.set_defaults(score_topics=True)
    args = parser.parse_args()

    print('gensim_lda.py')
    print(args)

    print("Creating a documents object")
    docs = Documents(journal_file=args.journal_file,
                     num_test=args.num_test,
                     data_dir=args.data_dir,
                     rebuild=args.rebuild,
                     keep_n=args.keep_n,
                     num_docs=args.num_docs,
                     shuffle=args.shuffle,
                     verbose=args.verbose)
    docs.fit()

    print("Build LDA model")
    lda = GensimLDA(docs=docs, n_workers=args.n_workers, verbose=args.verbose)
    performance = lda.fit(num_topics=args.num_topics,
            chunksizes=args.chunksizes,
            perplexity_threshold=args.threshold,
            evals_per_pass=args.evals_per_pass)
    
    print(performance)

    # save trained model to file
    lda.docs.train_bow = None
    lda.docs.test_bow = None
    lda.docs.train_keys = None
    lda.docs.test_keys = None
    pickle_it(lda, os.path.join(args.data_dir, "LDA_test_" + str(lda.num_test) + "_train_" + str(lda.num_train) + "_topics_" + str(lda.num_topics) + ".p"))

    print("Saving word topic probabilities...")
    lda._init_docs(docs)
    if args.score_topics:
        metric = "NPMI"
    else:
        metric = "none"
    lda.save_word_topic_probs(os.path.join(args.data_dir, "word_topic_probs.txt"), metric=metric)
    print("Saving topic terms...")
    lda.save_topic_terms(os.path.join(args.data_dir, "topic_terms.txt"), metric=metric)
    print("Saving document topic probabilities")
    lda.save_doc_topic_probs(docs.train_bow, docs.train_keys, os.path.join(args.data_dir, "train_document_topic_probs.txt"))

    
if __name__ == "__main__":
    main()
