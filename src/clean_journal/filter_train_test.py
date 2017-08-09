from __future__ import division, print_function, absolute_import
import os
import subprocess
import datetime as dt
import argparse


def to_date(x):
    return dt.datetime.fromtimestamp(x/1000)


def dif_days(x, y):
    d = x - y
    return d.days + (d.seconds / 60 / 60 / 24.)


def read_site_ids(filename):
    sites = {}
    with open(filename, "r") as f:
        for line in f:
            fields = line.split("\t")
            sites[fields[0]] = int(fields[1])
    return sites


def read_keys(filename):
    keys = []
    with open(filename, "r") as f:
        for line in f:
            fields = line.split("\t")
            keys.append(fields)
    return keys


def filter_authors(input_file, train_out, test_out, train_keys_file, test_keys_file):
    """
    assuming input file, train_keys, and test_keys are all sorted properly (site_id, author_id, journal_key, date)
    """
    # read in training and test keys
    train_keys = read_keys(train_keys_file)
    test_keys = read_keys(test_keys_file)
    print("head(train_keys)", train_keys[0:5])
    print("head(test_keys)", test_keys[0:5])
    
    # loop through the file, send training and testing sites to seperate files
    # remove any duplicate journals
    prev_keys = []
    train_ptr = 0
    test_ptr = 0
    with open(input_file, "r") as journal, open(train_out, "wb") as train, open(test_out, "wb") as test:
        for line in journal:
            if line == "\n":
                break

            fields = line.split("\t")
            keys = fields[0:4]

            if keys == prev_keys:
                # this journal is a repeat of the pervious, skip it
                continue

            # update prev keys for comparison next iteration
            prev_keys = keys
            site = fields[0]
            timestamp = int(fields[3])

            if train_ptr < len(train_keys) and train_keys[train_ptr][0:4] == keys:
                # put this line in train file
                relative_date = train_keys[train_ptr][4]
                fields[3] = str(relative_date)
                train.write('\t'.join(fields))
                train_ptr += 1
            elif test_ptr < len(test_keys) and test_keys[test_ptr][0:4] == keys:
                # put this line in test file
                relative_date = test_keys[test_ptr][4]
                fields[3] = str(relative_date)
                test.write('\t'.join(fields))
                test_ptr += 1
            else:
                # this is a skipped journal
                continue

    print("len(train_keys) =", len(train_keys), "and train_ptr =", train_ptr)
    print("len(test_keys) =", len(test_keys), "and test_ptr =", test_ptr)
            
            


def filter_hc(input_file, train_out, test_out, train_sites_file, test_sites_file): 
    # read in the train site ids and test site ids
    train_sites  = read_site_ids(train_sites_file)
    test_sites = read_site_ids(test_sites_file)
        
    # loop through the file, send training and testing sites to seperate files
    # remove any duplicate journals
    prev_keys = ""
    prev_site = ""
    prev_dist = ""
    with open(input_file, "r") as journal, open(train_out, "wb") as train, open(test_out, "wb") as test:
        for line in journal:
            if line == "\n":
                break
            
            fields = line.split("\t")
            keys = ''.join(fields[0:4])

            if keys == prev_keys:
                # this journal is a repeat of the previous, skip it
                continue

            # update prev keys for comparison next iteration
            prev_keys = keys

            num_words = len(fields[-1].split())
            if num_words < 10:
                continue
        
            site = fields[0]
            timestamp = int(fields[3])
            
            # again, like in the R script, make sure any journals with funky dates are ignored
            if timestamp < 0.86e+12 or timestamp > 1.5e+12:
                continue

            # direct to the appropriate output
            if site == prev_site:
                # this is another post from same site, we know where it goes
                if prev_dest == "train":
                    relative_date = dif_days(to_date(timestamp), to_date( train_sites[site]))
                    fields[3] = str(relative_date)
                    train.write('\t'.join(fields))
                elif prev_dest == "test":
                    relative_date = dif_days(to_date(timestamp), to_date(test_sites[site]))
                    fields[3] = str(relative_date)
                    test.write('\t'.join(fields))
                else:
                    continue
            else:
                # this is a new site
                prev_site = site
                # where should this new site go?
                if site in train_sites:
                    prev_dest = "train" # if we see this site again, it goes in train
                    relative_date = dif_days(to_date(timestamp), to_date( train_sites[site]))
                    fields[3] = str(relative_date)
                    train.write('\t'.join(fields))
                elif site in test_sites:
                    prev_dest = "test" # if we see this site again, it goes in test
                    relative_date = dif_days(to_date(timestamp), to_date(test_sites[site]))
                    fields[3] = str(relative_date)
                    test.write('\t'.join(fields))
                else:
                    prev_dest = "skip" # if we see this site again, skip it
                    continue



def main():
    parser = argparse.ArgumentParser(description='This program filters a journal file to on keep certain sites and removes duplicate journals..')
    parser.add_argument('--data_dir', type=str, help='Data directory where input and output files should be/go.')
    parser.add_argument('--input_file', type=str, help='Name of file to read input journals from.')
    parser.add_argument('--train_keys', type=str, help='List of keys for training.')
    parser.add_argument('--test_keys', type=str, help='List of keys for testing.')
    parser.add_argument('--train_out', type=str, help='training file written out.')
    parser.add_argument('--test_out', type=str, help='testing file written out.')
    parser.add_argument('--method', type=str, help='method authors or hc.')
    args = parser.parse_args()

    print('filter_train_test.py')
    print(args)

    input_file = os.path.join(args.data_dir, args.input_file)
    train_out = os.path.join(args.data_dir, args.train_out)
    test_out = os.path.join(args.data_dir, args.test_out)
    train_keys = os.path.join(args.data_dir, args.train_keys)
    test_keys = os.path.join(args.data_dir, args.test_keys)
    if args.method == "hc":
        filter_hc(input_file, train_out, test_out, train_keys, test_keys)
    elif args.method == "authors":
        filter_authors(input_file, train_out, test_out, train_keys, test_keys)
    else:
        raise ValueError("method must be hc or authors")

if __name__ == "__main__":
    main()
