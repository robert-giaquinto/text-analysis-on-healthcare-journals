from __future__ import division, print_function, absolute_import
import os
import subprocess
import datetime as dt


def main():
    #data_dir = "/home/srivbane/shared/caringbridge/data/clean_journals/"
    data_dir = "/home/srivbane/shared/caringbridge/data/word_embeddings/"
    journal_file = "clean_journals_for_word2vec.txt"
    out_dir = "/home/srivbane/shared/caringbridge/data/word_embeddings/"
    keep_sites_file = "keep_sites.txt"
    keep_out = "clean_journals_for_word2vec_filtered.txt"

    # Make sure that the journal file is sorted
    check_sorted_cmd = """/bin/bash -c "sort -nc %s -t$'\t' -k1,4 -S %s" """ % (data_dir + journal_file, "80%")
    try:
        subprocess.check_call(check_sorted_cmd, shell=True)
        print("File aleady sorted properly.")
    except subprocess.CalledProcessError as e:
        print("Sorting file.")
        cmd = """/bin/bash -c "sort -n %s -t$'\t' -k1,4 -o %s -S %s -T /home/srivbane/shared/caringbridge/data/tmp" """ % (data_dir + journal_file, data_dir + journal_file, "80%")
        subprocess.call(cmd, shell=True)

    # read in the keep site ids and test site ids
    keep_sites = []
    with open(out_dir + keep_sites_file, "r") as keep:
        for line in keep:
            fields = line.split("\t")
            keep_sites.append(fields[0])
    keep_sites = set(keep_sites)

    # loop through the file
    # remove any duplicate journals
    prev_keys = ""
    prev_site = ""
    prev_dist = ""
    with open(data_dir + journal_file, "r") as journal, open(out_dir + keep_out, "wb") as keep:
        for line in journal:
            if line == "\n":
                break
            
            fields = line.split("\t")
            keys = ' '.join(fields[0:4])
            
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
                if prev_dest == "keep":
                    keep.write(line)
                else:
                    continue
            else:
                # this is a new site
                prev_site = site
                # where should this new site go?
                if site in keep_sites:
                    prev_dest = "keep" # if we see this site again, it goes in keep
                    keep.write(line)
                else:
                    prev_dest = "skip" # if we see this site again, skip it
                    continue

if __name__ == "__main__":
    main()
