from __future__ import division, print_function
from math import ceil
import os

# TODO should this be a file for holding general utilities functions?
# in which case it shouldn't be in the parse journal file...

def split_json_file(filename, n_splits, infile_len=None):
    """
    Split a big file into multiple shards

    Args:
        filename: name of the file to split up
        n_splits: how many shards should filename be broken into
        infile_len: how many lines are in the file, you can pass this in
                    to speed things up
    Return: None, just create a directory in the same folder as 
            the filename containing all the shards
    """
    if infile_len is None:
        in_file_len = count_lines(filename)

    # when copying lines of 'filename' to a new file, need to know when to begin copying the lines to a new file 
    lines_per_shard = ceil(infile_len / n_splits)
    split_points = []
    line_ct = 0
    while line_ct < infile_len -  lines_per_shard:
        line_ct += lines_per_shard
        split_points.append(line_ct)

    # create a new directory containing all the file shards
    file_dir = '/'.join(filename.split('/')[0:-1])
    file_prefix = filename.split('/')[-1].replace('.json', '')
    shard_dir = os.path.join(file_dir, file_prefix + '_shards/')
    if not os.path.isdir(shard_dir):
        os.makedirs(shard_dir)
    
    # loop through filename given, copy lines_per_shard lines to each new file
    outfile_ct = 0
    outfile = open(shard_dir + file_prefix + '_01_of_' + str(n_splits) + '.json', 'wb')
    with open(filename, 'r') as fin:
        for i, line in enumerate(fin):
            if i in split_points:
                # split point reached, close current file, open a new one
                outfile.close()
                outfile_ct +=1
                outfile = open(shard_dir + file_prefix + '_' + str(outfile_ct + 1).zfill(2) + '_of_' + str(n_splits) + '.json', 'wb')
                outfile.write(line)
            else:
                # keep writing to the same file that's currently open
                outfile.write(line)
    outfile.close()
    

def count_lines(filename):
    with open(filename, "r") as fin:
        ct = sum(1 for _ in fin)
    return ct
