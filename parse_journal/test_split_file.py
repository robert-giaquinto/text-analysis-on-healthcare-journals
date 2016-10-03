from __future__ import division, print_function
import unittest
import os
from math import floor
from utilities import split_file, count_lines

class TestSplitFile(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        #create a file to load in and run tests on
        if not os.path.exists("./test_data"):
            os.makedirs("./test_data")
        with open("./test_data/test_journal.json", "wb") as f:
            for s in range(4):
                for j in range(3):
                    if j > 0:
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "0", "title" : "TITLE", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    else:
                        line ='{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "1", "title" : "TITLE", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    f.write(line)
    @classmethod
    def tearDownClass(cls):
        os.remove('./test_data/test_journal.json')
        os.rmdir('./test_data')

    def setUp(self):
        # split the file before each test, don't worry about the names of the files
        _ = split_file('./test_data/test_journal.json', n_splits=2, infile_len=12)

    def tearDown(self):
        # remove the file shards created
        filenames = os.listdir('./test_data/test_journal_shards/')
        for f in filenames:
            os.remove('./test_data/test_journal_shards/' + f)
        os.rmdir('./test_data/test_journal_shards')
        
    def test_split_count_files(self):
        n_splits = 2
        expected_filenames = ['test_journal_' + str(i+1).zfill(2) + '_of_' + str(n_splits) + '.json' for i in range(n_splits)]
        actual_filenames = os.listdir('./test_data/test_journal_shards')
        self.assertItemsEqual(expected_filenames, actual_filenames)

    def test_split_count_file_lengths(self):
        n_splits = 2
        original_length = count_lines('./test_data/test_journal.json')
        expected_sizes = [int(floor(1.0 * original_length / n_splits)) for i in range(n_splits - 1)]
        expected_sizes += [original_length - sum(expected_sizes)]

        filenames = os.listdir('./test_data/test_journal_shards')
        actual_sizes = []
        for f in filenames:
            actual_sizes.append(count_lines('./test_data/test_journal_shards/' + f))
        actual_sizes = sorted(actual_sizes)
        self.assertItemsEqual(expected_sizes, actual_sizes)

    def test_count_lines(self):
        expected = count_lines('./test_data/test_journal.json')
        actual = 12
        self.assertEqual(expected, actual)
