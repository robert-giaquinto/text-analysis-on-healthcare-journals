from __future__ import division, print_function, absolute_import
import unittest
import os
from math import floor
from src.collect_journal_keys import KeyCollector


class TestCollecter(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_collector_data"):
            os.makedirs("./test_collector_data")

        for siteId in range(4):
            site_dir = "./test_collector_data/" + str(siteId)
            if not os.path.exists(site_dir):
                os.makedirs(site_dir)

            for journalId in range(4):
                filename = os.path.join(site_dir, '_'.join([str(siteId), '0', str(journalId), '20161010']))
                with open(filename, 'wb') as fout:
                    fout.write('body' + str(siteId) + str(journalId))

    def remove_test_data(self):
        for siteId in range(4):
            site_dir  = os.path.join('./test_collector_data/', str(siteId))
            filenames = os.listdir(site_dir)
            for filename in filenames:
                os.remove(os.path.join(site_dir, filename))
            os.rmdir(site_dir)

        os.remove('./test_collector_data/test_output.txt')
        os.rmdir('./test_collector_data')

    def test_key_collector_count(self):
        self.create_test_data()
        kc = KeyCollector(input_dir='./test_collector_data', output_filename='./test_collector_data/test_output.txt')
        kc.collect_keys()

        with open('./test_collector_data/test_output.txt', 'r') as fin:
            actual = 0
            for line in fin:
                actual += 1

        self.assertEqual(actual, 16)
        self.remove_test_data()

    def test_key_collector_content(self):
        self.create_test_data()
        kc = KeyCollector(input_dir='./test_collector_data', output_filename='./test_collector_data/test_output.txt')
        kc.collect_keys()

        actual = []
        with open('./test_collector_data/test_output.txt', 'r') as fin:
            for line in fin:
                actual.append(line.replace('\n', ''))

        expected = []
        for siteId in range(4):
            for journalId in range(4):
                expected.append('\t'.join([str(siteId), '0', str(journalId), '20161010']))

        self.assertEquals(actual, expected)
        self.remove_test_data()

