from __future__ import division, print_function, absolute_import
import unittest
import os
import numpy as np

from src.topic_model.lda_utils import *


class TestLDAUtils(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_lda_data"):
            os.makedirs("./test_lda_data")

        keys = "0,0,0,0,"
        keys2 = "1,1,1,1,"
        with open("./test_lda_data/input.txt", "wb") as fout:
            fout.write(keys + "0.1, 0.3, 0.05, 0.55\n")
            fout.write(keys + "0.55, 0.1, 0.3, 0.05\n")
            fout.write(keys + "0.05, 0.55, 0.1, 0.3\n")
            fout.write(keys + "0.3, 0.05, 0.55, 0.1\n")
            fout.write(keys2 + "0.26, 0.24, 0.25, 0.25\n")
            fout.write(keys2 + "0.25, 0.26, 0.24, 0.25\n")
            fout.write(keys2 + "0.25, 0.25, 0.26, 0.24\n")
            fout.write(keys2 + "0.24, 0.25, 0.25, 0.26\n")


    def remove_test_data(self):
        for f in os.listdir('./test_lda_data/'):
            os.remove('./test_lda_data/' + f)
        os.rmdir('./test_lda_data')

    def test_lda_utils_kl(self):
        p = np.array([0.1, 0.3, 0.05, 0.55])
        actual = round(kl_div(p), 5)
        expected = 0.45625
        self.assertEqual(actual, expected)

    def test_lda_utils_compute_doc_topic_stats(self):
        self.create_test_data()
        compute_doc_topic_stats("./test_lda_data/input.txt", "./test_lda_data/stats.txt")
        actual = []
        with open("./test_lda_data/stats.txt", 'r') as fin:
            for line in fin:
                line = line.replace("\n", "")
                fields = line.split(",")
                kl = round(float(fields[-1]), 5)
                actual.append(','.join(fields[0:-1]) + "," + str(kl))

        keys = "0,0,0,0,"
        keys2 = "1,1,1,1,"
        expected = [keys + "3,0.45625", keys + "0,0.45625", keys + "1,0.45625", keys + "2,0.45625",
                    keys2 + "0,0.00058", keys2 + "1,0.00058", keys2 + "2,0.00058", keys2 + "3,0.00058"]
        self.remove_test_data()
        self.assertItemsEqual(actual, expected)

    def test_lda_utils_select(self):
        self.create_test_data()
        compute_doc_topic_stats("./test_lda_data/input.txt", "./test_lda_data/stats.txt")
        actual = select_best_docs_per_topic("./test_lda_data/stats.txt", 1)
        keys = "0,0,0,0".split(",")
        expected = [('3',keys), ('0',keys), ('1',keys), ('2', keys)]
        self.remove_test_data()
        self.assertItemsEqual(actual, expected)
        
