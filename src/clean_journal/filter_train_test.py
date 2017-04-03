from __future__ import division, print_function, absolute_import
import os
import subprocess
import datetime as dt


def to_date(x):
    return dt.datetime.fromtimestamp(x/1000)

def dif_days(x, y):
    d = x - y
    return d.days + (d.seconds / 60 / 60 / 24.)


def main():
    #data_dir = "/home/srivbane/shared/caringbridge/data/dev/clean_journals/"
    data_dir = "/home/srivbane/shared/caringbridge/data/clean_journals/"
    #journal_file = "test.txt"
    journal_file = "clean_journals_hom_names.txt"
    train_file = "train_sites.txt"
    train_out = "train_10k_sites_known_hc.txt"
    test_file = "test_sites.txt"
    test_out = "test_10k_sites_known_hc.txt"

    # Make sure that the journal file is sorted
    check_sorted_cmd = """/bin/bash -c "sort -nc %s -t$'\t' -k1,4 -S %s" """ % (data_dir + journal_file, "80%")
    try:
        subprocess.check_call(check_sorted_cmd, shell=True)
        print("File aleady sorted properly.")
    except subprocess.CalledProcessError as e:
        print("Sorting file.")
        cmd = """/bin/bash -c "sort -n %s -t$'\t' -k1,4 -o %s -S %s -T /home/srivbane/shared/caringbridge/data/tmp" """ % (data_dir + journal_file, data_dir + "sorted_" + journal_file, "80%")
        subprocess.call(cmd, shell=True)
        journal_file = "sorted_" + journal_file

    # read in the train site ids and test site ids
    train_sites = {}
    with open(data_dir + train_file, "r") as train:
        for line in train:
            fields = line.split("\t")
            train_sites[fields[0]] = int(fields[1])

    test_sites = {}
    with open(data_dir + test_file, "r") as test:
        for line in test:
            fields = line.split("\t")
            test_sites[fields[0]] = int(fields[1])

        
    # loop through the file, send training and testing sites to seperate files
    # remove any duplicate journals
    prev_keys = ""
    prev_site = ""
    prev_dist = ""
    with open(data_dir + journal_file, "r") as journal, open(data_dir + train_out, "wb") as train, open(data_dir + test_out, "wb") as test:
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

    # sort the final results by time
    print("Sorting final results by time")
    cmd = """/bin/bash -c "sort -n %s -t$'\t' -k4,4 -o %s -S %s -T /home/srivbane/shared/caringbridge/data/tmp" """ % (data_dir + train_out, data_dir + train_out, "80%")
    subprocess.call(cmd, shell=True)
    cmd = """/bin/bash -c "sort -n %s -t$'\t' -k4,4 -o %s -S %s -T /home/srivbane/shared/caringbridge/data/tmp" """ % (data_dir + test_out, data_dir + test_out, "80%")
    subprocess.call(cmd, shell=True)



if __name__ == "__main__":
    main()
