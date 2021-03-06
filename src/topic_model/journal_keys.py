from __future__ import division, print_function, absolute_import
import os


class JournalKeys(object):
    """
    Iterable: on each iteration, return journal keys in a list,
    one list for each journal.

    Process one cleaned journal at a time using generators, never
    load the entire corpus into RAM.

    Using an iterable so that memory isn't a concern, and
    Gensim vocabulary and BOW building tools work well
    with iterables.
    """
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        """
        Input file is assumed to be tab seperated, containg these fields:
        siteId    userId    journalId    createdAt    [space seperated tokens from text]

        Args: None, load data one at a time from self.filename
        Return: None. Yield the tokens in a list, one journal at a time.
        """
        with open(self.filename, "r") as fin:
            for line in fin:
                fields = line.replace("\n", "").split("\t")
                keys = fields[0:-1]
                yield keys


def main():
    """
    Simple example of how to use this class
    """
    jt = JournalKeys(filename = '/home/srivbane/shared/caringbridge/data/dev/clean_journals/cleaned_journals.txt')
    print("Here are top the top lines of the cleaned journals in the dev folder (only showing keys)")
    for i, keys in enumerate(jt):
        if i > 5:
            break
        print(', '.join(sorted(keys)))

if __name__ == "__main__":
    main()
