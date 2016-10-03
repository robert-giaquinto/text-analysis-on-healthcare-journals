from __future__ import division, print_function
from math import ceil
import os

# TODO should this be a file for holding general utilities functions?
# in which case it shouldn't be in the parse journal file...

def split_file(filename, n_splits, infile_len=None):
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
        infile_len = count_lines(filename)

    # when copying lines of 'filename' to a new file, need to know when to begin copying the lines to a new file 
    lines_per_shard = int(ceil(1.0 * infile_len / n_splits))
    split_points = []
    line_ct = 0
    while line_ct < infile_len -  lines_per_shard:
        line_ct += lines_per_shard
        split_points.append(line_ct)

    # create a new directory containing all the file shards
    file_dir = '/'.join(filename.split('/')[0:-1])
    file_prefix, file_ext = os.path.splitext(filename.split('/')[-1])
    shard_dir = os.path.join(file_dir, file_prefix + '_shards/')
    if not os.path.isdir(shard_dir):
        os.makedirs(shard_dir)
    
    # loop through filename given, copy lines_per_shard lines to each new file
    outfile_ct = 1
    out_name = shard_dir + file_prefix + '_01_of_' + str(n_splits) + file_ext
    filenames = [out_name]
    outfile = open(out_name, 'wb')
    with open(filename, 'r') as fin:
        for i, line in enumerate(fin):
            if i in split_points:
                # split point reached, close current file, open a new one
                outfile.close()
                outfile_ct +=1
                out_name = shard_dir + file_prefix + '_' + str(outfile_ct).zfill(2) + '_of_' + str(n_splits) + file_ext
                outfile = open(out_name, 'wb')
                outfile.write(line)
                filenames.append(out_name)
            else:
                # keep writing to the same file that's currently open
                outfile.write(line)
    outfile.close()
    return filenames
    

def count_lines(filename):
    with open(filename, "r") as fin:
        ct = sum(1 for _ in fin)
    return ct


# functions for pickling with multiprocessing (to pass object
# instances to a multiprocessor pool
def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)
