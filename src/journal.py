from __future__ import division, print_function, absolute_import


class Journal(object):
    """
    Just a simple structure to hold journal information
    """
    def __init__(self, siteId=None, userId=None, journalId=None, createdAt=None, body=None):
        self.siteId = siteId
        self.userId = userId
        self.journalId = journalId
        self.createdAt = createdAt
        self.body = body
        self.features = []
        self.tokenized = False

    def __repr__(self):
        return "Journal Object"

    def __str__(self):
        return "\nsiteId: " + str(self.siteId) +\
            "\nuserId: " + str(self.userId) +\
            "\njournalId: " + str(self.journalId) +\
            "\ncreatedAt: " + str(self.createdAt) +\
            "\nbody: " + ' '.join(self.body) if self.tokenized else self.body
