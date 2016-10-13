from __future__ import division, print_function, absolute_import
import unittest
import os

from src.topic_model.documents import Documents
from src.utilities import count_lines


class TestDocuments(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_document_data"):
            os.makedirs("./test_document_data")
        with open("./test_document_data/input.tsv", "wb") as fin:
            for siteId in range(5):
                for journalId in range(4):
                    line = "%d\t0\t%d\t20161011\t%s" % (siteId, journalId, "this is some text\n")
                    fin.write(line)

    def remove_test_data(self):
        for f in os.listdir('./test_document_data/input_shards/'):
            os.remove('./test_document_data/input_shards/' + f)
        os.rmdir('./test_document_data/input_shards/')
        for f in os.listdir('./test_document_data/'):
            os.remove('./test_document_data/' + f)
        os.rmdir('./test_document_data')

    def test_documents_split_counts(self):
        self.create_test_data()
        docs = Documents('./test_document_data/input.tsv', num_test=5, data_dir='./test_document_data', keep_n=4, rebuild=True, num_docs=20, verbose=False)
        docs.fit()
        train_sz = count_lines('./test_document_data/input_shards/input_01_of_2.tsv')
        test_sz = count_lines('./test_document_data/input_shards/input_02_of_2.tsv')
        actual = [train_sz, test_sz]
        expected = [15, 5]
        self.assertItemsEqual(actual, expected)
        self.remove_test_data()

