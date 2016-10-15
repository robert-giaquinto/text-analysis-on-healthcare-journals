from __future__ import division, print_function, absolute_import
import unittest
import os
from math import floor
from src.utilities import split_file, count_lines
from src.parse_journal.parse_worker import JournalParsingWorker
import json


class TestParsing(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_parse_data"):
            os.makedirs("./test_parse_data")
        with open("./test_parse_data/test_input.json", "wb") as f:
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
        os.remove('./test_parse_data/test_input.json')
        os.remove('./test_parse_data/parsed_test_input.txt')
        os.rmdir('./test_parse_data')

    def test_worker_check_skip(self):
        # create a few examples that should be skipped
        isdraft = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDraft" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        isdraft = json.loads(isdraft)

        isdeleted = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        isdeleted = json.loads(isdeleted)

        default = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "0", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "This CaringBridge site was created just recently. Please visit again soon for a journal update.",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        default = json.loads(default)

        nobody = """{ "_id" : { "$oid" : "x" }, "siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "0", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3",
            "updatedAt" : { "$date" : 371412342000 }, "createdAt" : { "$date" : 1371412342000 } }"""
        nobody = json.loads(nobody)

        nodate = """{"siteId" : 0, "journalId" : 0,
            "userId" : 0, "isDeleted" : "0", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY." }"""
        nodate = json.loads(nodate)

        nositeid = """{"journalId" : 0,
            "userId" : 0, "isDeleted" : "1", "title" : "TITLE", "amps" : [],
            "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY." }"""
        nositeid = json.loads(nositeid)

        # is the number of skips equal to 5?
        worker = JournalParsingWorker(input_path=None, output_dir=None, verbose=False)
        expected = worker.check_skip(isdraft) + worker.check_skip(isdeleted) + worker.check_skip(default) + worker.check_skip(nobody) + worker.check_skip(nodate) + worker.check_skip(nositeid)
        self.assertEqual(expected, 6)

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

    def test_worker_parse_file_content(self):
        self.create_test_data()
        worker = JournalParsingWorker(input_path='./test_parse_data/test_input.json', output_dir='./test_parse_data/', verbose=False)
        worker.parse_file()

        expected = []
        for siteId in range(4):
            for journalId in range(4):
                if journalId == 0:
                    expected.append(str(siteId) + '\t0\t' + str(journalId) + '\t1371412342000\tTITLE BODY' + str(siteId) + str(journalId))
                elif journalId == 1:
                    expected.append(str(siteId) + '\t0\t' + str(journalId) + '\t1371412342000\tBODY'+ str(siteId) + str(journalId))
                elif journalId == 2:
                    expected.append(str(siteId) + '\t0\t-1' + '\t1371412342000\tBODY'+ str(siteId) + str(journalId))
                elif journalId == 3:
                    expected.append(str(siteId) + '\t-1\t' + str(journalId) + '\t1371412342000\tBODY'+ str(siteId) + str(journalId))

        actual = []
        with open('./test_parse_data/parsed_test_input.txt', 'r') as fin:
            for line in fin:
                actual.append(line.replace("\n", ""))

        self.assertItemsEqual(actual, expected)
        self.remove_test_data()


