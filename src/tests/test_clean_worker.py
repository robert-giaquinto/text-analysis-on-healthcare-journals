from __future__ import division, print_function, absolute_import
import unittest
import os
import re
from src.journal import Journal
from src.clean_journal.clean_worker import JournalCleaningWorker


class TestJournalsTM(unittest.TestCase):
    """
    Test cases for cleaning the journal entries
    for topic modeling
    """
    def test_journal_clean_html(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = "<html><body>this is the first<br>testcase</body></html>"
        j = Journal(body=body)
        expected = " this is the first testcase "
        actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, tokenize=False)
        self.assertEqual(actual.body, expected)

    def test_journal_clean_whitespace(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = "this\thas\nwhite\rspace"
        j = Journal(body=body)
        expected = "this has white space"
        actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, tokenize=False)
        self.assertEqual(actual.body, expected)

    def test_journal_clean_hyphens(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = "this-is 3-rd testcase-4 and/or more"
        j = Journal(body=body)
        expected = "this is 3 rd testcase 4 and or more"
        actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, rm_punct=False, tokenize=False)
        self.assertEqual(actual.body, expected)

    def test_journal_clean_special_patterns(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = r"in the year 2020 at 8:00 am, i will have $5, $5.25 or $5,000,000 but not $6,000,000. yet, which is 50% or 50.50% nonesense."
        j = Journal(body=body)
        expected = "in the year _year_ at _time_ am, i will have _dollars_ _dollars_ or _dollars_ but not _dollars_ yet, which is _percent_ or _percent_ nonesense."
        journal_actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, rm_punct=False, tokenize=False)
        actual = re.sub("\s+", " ", journal_actual.body)
        self.assertEqual(actual, expected)

    def test_journal_clean_contractions(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = "I'm won't can't it's let's he's how's that's there's what's where's when's who's why's ya'll y'all you're he'd that'd they'll I'll Johnnies' Robert's don't shouldn't should've I've"
        j = Journal(body=body)
        expected = "I am will not can not it is let us he is how is that is there is what is where is when is who is why is you all you all you are he would that would they will I will Johnnies has Robert has do not should not should have I have"
        journal_actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, tokenize=False)
        actual = re.sub("\s+", " ", journal_actual.body)
        self.assertEqual(actual, expected)

    def test_journal_clean_special_chars(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = "&quote;this &amp; that&quote;"
        j = Journal(body=body)
        expected = " this   that "
        journal_actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, tokenize=False)
        self.assertEqual(journal_actual.body, expected)

    def test_journal_clean_punct_and_numbers(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = """'Here "we' go, but! not? quite@ this # $ok until 1*1=10 (YES!) 2 - 2 = 0 smit7982@umn.edu: is [ok]"""
        j = Journal(body=body)
        expected = " Here we go but not quite this ok until YES smit umn edu is ok "
        journal_actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False, tokenize=False)
        actual = re.sub("\s+", " ", journal_actual.body)
        self.assertEqual(actual, expected)

    def test_journal_clean_punct_and_numbers(self):
        jrnls = JournalCleaningWorker(clean_method="topic")
        body = " spaces   (ok) yes/no 100.0 ! \n MORE\tend"
        j = Journal(body=body)
        expected = ['spaces', 'ok', 'yes', 'no', 'MORE', 'end']
        actual = jrnls.clean_journal_for_topic_modeling(j, as_ascii=False, rm_stopwords=False, lemmatize=False)
        self.assertItemsEqual(actual.body, expected)
