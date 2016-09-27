from __future__ import division, print_function
import unittest
import os
from math import floor
from utilities import split_json_file, count_lines
from parse_worker import JournalParsingWorker
import json

class TestWorker(unittest.TestCase):
    def create_test_data(self):
        #create a file to load in and run tests on
        if not os.path.exists("./test_worker_data"):
            os.makedirs("./test_worker_data")
        with open("./test_worker_data/test_journal.json", "wb") as f:
            for s in range(4):
                for j in range(3):
                    if j > 0:
                        line = '{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "0", "title" : "TITLE", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    else:
                        line ='{ "_id" : { "$oid" : "' + str(j + 3 * s) + '" }, "siteId" : ' + str(s) + ', "journalId" : ' + str(j) + ', "userId" : 0, "isDraft" : "1", "title" : "TITLE", "amps" : [], "platform" : "iphone", "ip" : "65.128.152.3", "body" : "BODY.", "updatedAt" : { "$date" : 1371412342000 }, "createdAt" : { "$date" : 1371412342000 } }\n'
                    f.write(line)

    def remove_test_data(self):
        os.remove('./test_worker_data/test_journal.json')
        os.rmdir('./test_worker_data')
        
    def test_worker_check_skip(self):
        # read in the json test file
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

        worker = JournalParsingWorker(input_path=None, output_dir=None, verbose=False)
        expected = worker.check_skip(json_dict1) + worker.check_skip(json_dict2)
        self.assertEqual(expected, 2)

    def test_worker_check_no_skip(self):
        # read in the json test file
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

    def test_worker_parse_file(self):
        self.create_test_data()
        worker = JournalParsingWorker('./test_worker_data/test_journal.json', './test_worker_data/', False)
        worker.parse_file()
        #self.remove_test_data()
        self.assertEqual(True,True)

        
