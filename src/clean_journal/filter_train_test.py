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


def main():
    parser = argparse.ArgumentParser(description='This program filters a journal file to on keep certain sites and removes duplicate journals..')
    parser.add_argument('--data_dir', type=str, help='Data directory where input and output files should be/go.')
    parser.add_argument('--input_file', type=str, help='Name of file to read input journals from.')
    parser.add_argument('--train_sites', type=str, help='List of sites for training.')
    parser.add_argument('--test_sites', type=str, help='List of sites for testing.')
    parser.add_argument('--train_out', type=str, help='training file written out.')
    parser.add_argument('--test_out', type=str, help='testing file written out.')
    args = parser.parse_args()

    print('filter_train_test.py')
    print(args)

    input_file = os.path.join(args.data_dir, args.input_file)
    train_out = os.path.join(args.data_dir, args.train_out)
    test_out = os.path.join(args.data_dir, args.test_out)
    train_file = os.path.join(args.data_dir, args.train_sites)
    test_file = os.path.join(args.data_dir, args.test_sites)
    
    # read in the train site ids and test site ids
    train_sites = {}
    with open(train_file, "r") as train:
        for line in train:
            fields = line.split("\t")
            train_sites[fields[0]] = int(fields[1])

    test_sites = {}
    with open(test_file, "r") as test:
        for line in test:
            fields = line.split("\t")
            test_sites[fields[0]] = int(fields[1])

        
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


if __name__ == "__main__":
    main()
