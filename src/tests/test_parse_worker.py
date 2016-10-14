from __future__ import division, print_function, absolute_import
import unittest
import os
from math import floor
from src.utilities import split_file, count_lines
from src.parse_journal.parse_worker import JournalParsingWorker
import json


class TestWorker(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_worker_data"):
            os.makedirs("./test_worker_data")
        with open("./test_worker_data/test_journal.json", "wb") as f:
            for s in range(4):
                for j in range(6):
                    if j == 0:
                        # normal data, no delete flags, all fields exist
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "0", "title" : "TITLE", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY' + str(s) + str(j) + '", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    elif j == 1:
                        # no titles
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "0", "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY' + str(s) + str(j) + '", "createdAt" : { "$date" : 1371412342000 } }\n'
                    elif j == 2:
                        # no journal id
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "userId" : 0, "isDraft" : "0", "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY' + str(s) + str(j) + '", "createdAt" : { "$date" : 1371412342000 } }\n'
                    elif j == 3:
                        # no user id
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "isDraft" : "0", "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY' + str(s) + str(j) + '", "createdAt" : { "$date" : 1371412342000 } }\n'
                    elif j == 4:
                        # default journal entry, should be removed
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "0", "platform" : "iphone", "ip" : "65.128.152.3", "body" : "This CaringBridge site was created just recently. Please visit again soon for a journal update.", "createdAt" : { "$date" : 1371412342000 } }\n'
                    else:
                        # delete flag
                        line ='{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "1", "title" : "TITLE' + str(s) + str(j) + '", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY' + str(s) + str(j) + '", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    f.write(line)

    def remove_test_data(self):
        for siteId in range(4):
            filenames = os.listdir('./test_worker_data/' + str(siteId))
            for fname in filenames:
                os.remove('./test_worker_data/' + str(siteId) + '/' + fname)
            os.rmdir('./test_worker_data/' + str(siteId))
        os.remove('./test_worker_data/test_journal.json')
        os.rmdir('./test_worker_data')

    def test_worker_check_skip(self):
        # create a few examples that should be skipped
        text1 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDraft" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict1 = json.loads(text1)

        text2 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict2 = json.loads(text2)

        text3 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "This CaringBridge site was created just recently. Please visit again soon for a journal update.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict3 = json.loads(text3)

        text4 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict4 = json.loads(text4)

        text5 = """{"siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY." }"""
        json_dict5 = json.loads(text5)

        # is the number of skips equal to 5?
        worker = JournalParsingWorker(input_path=None, output_dir=None, verbose=False)
        expected = worker.check_skip(json_dict1) + worker.check_skip(json_dict2) + worker.check_skip(json_dict3) + worker.check_skip(json_dict4) + worker.check_skip(json_dict5)
        self.assertEqual(expected, 5)

    def test_worker_check_no_skip(self):
        # give some examples that shouldn't be skipped
        text1 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDraft" : "0", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict1 = json.loads(text1)

        text2 = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "0", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        json_dict2 = json.loads(text2)

        worker = JournalParsingWorker(input_path=None, output_dir=None, verbose=False)
        expected = worker.check_skip(json_dict1) + worker.check_skip(json_dict2)
        self.assertEqual(expected, 0)

    def test_worker_parse_file_filenames(self):
        self.create_test_data()
        worker = JournalParsingWorker(input_path='./test_worker_data/test_journal.json', output_dir='./test_worker_data/', verbose=False)
        worker.parse_file()
        actual = []
        expected = []
        for siteId in range(4):
            actual += os.listdir('./test_worker_data/' + str(siteId))
            for journalId in range(4):
                if journalId < 2:
                    expected.append(str(siteId) + '_0_' + str(journalId) + '_1371412342000')
                elif journalId == 2:
                    expected.append(str(siteId) + '_0_-1_1371412342000') # one missing journalId per site
                elif journalId == 3:
                    expected.append(str(siteId) + '_-1_' + str(journalId) + '_1371412342000') # one missing user per site

        self.remove_test_data()
        self.assertItemsEqual(expected, actual)

    def test_worker_parse_file_content(self):
        self.create_test_data()
        worker = JournalParsingWorker(input_path='./test_worker_data/test_journal.json', output_dir='./test_worker_data/', verbose=False)
        worker.parse_file()
        actual = []
        expected = []
        for siteId in range(4):
            filenames = os.listdir('./test_worker_data/' + str(siteId))
            for fname in filenames:
                with open('./test_worker_data/' + str(siteId) + '/' + fname, 'r') as fin:
                    actual.append(fin.readline().replace('\n', '').strip())

            for journalId in range(4):
                if journalId == 0:
                    expected.append('TITLE BODY' + str(siteId) + str(journalId))
                else:
                    expected.append('BODY' + str(siteId) + str(journalId))

        self.remove_test_data()
        self.assertItemsEqual(expected, actual)


