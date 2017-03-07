from __future__ import print_function, division, absolute_import
import os


def gather_health_conditions(hc_file, hc_out_keys_file):
    hc_keys = {}
    author_hc = {}

    with open(hc_file, "r") as f:
        for line in f:
            if line == "\n":
                break
            
            fields = line.replace("\n", "").split("\t")
            hc = fields[1]
            if hc == "custom":
                continue

            if hc not in hc_keys:
                hc_keys[hc] = len(hc_keys) + 1

            author = fields[0]
            author_hc[author] = hc_keys[hc]

    with open(hc_out_keys_file, "wb") as f:
        for k, v in sorted(hc_keys.items(), key=lambda x: x[1]):
            f.write(k + "\t" + str(v) + "\n")

    return author_hc


def write_apt_data(keys_file, cdtm_file, out_file, author_hc=None):
    kfile = open(keys_file, "r")
    
    docs_in_step_remaining = 0
    with open(cdtm_file, "r") as cfile, open(out_file, "wb") as ofile:
        for i, line in enumerate(cfile):
            if line == "\n":
                break
            
            if i == 0:
                ofile.write(line)
                continue
            
            if docs_in_step_remaining == 0:
                print("Time step:", line.replace("\n", ""))
                ofile.write(line)
                docs_in_step_remaining -= 1
                continue

            if docs_in_step_remaining == -1:
                docs_in_step_remaining = int(line.replace("\n", ""))
                ofile.write(line)
                continue

            # append author keys and LDA-C formatted data
            keys_line = kfile.readline()
            author = keys_line.split("\t")[0]
            if author_hc is not None:
                # lookup health condition of author
                author_info = author_hc[author]
            else:
                # using unique author id as author information
                author_info = author

            ofile.write(str(author_info) + " " + line)
            docs_in_step_remaining -= 1

    kfile.close()

def main():
    data_dir = "/home/srivbane/shared/caringbridge/data/apt/"
    hc_file = "/home/srivbane/shared/caringbridge/data/clean_journals/health_condition.txt"
    hc_out_keys_file = data_dir + "health_condition_key.txt"
    train_keys_file = data_dir + "train_10k_sites_known_hc_key.txt"
    train_cdtm_file = data_dir + "train.dat"
    train_out_file = data_dir + "train-apt.dat"
    holdout_keys_file = data_dir + "test_10k_sites_known_hc_key.txt"
    holdout_cdtm_file = data_dir + "holdout.dat"
    holdout_out_file = data_dir + "holdout-apt.dat"

    # process health conditions
    print("Gathering author health condition lookup table")
    author_hc = gather_health_conditions(hc_file, hc_out_keys_file)

    # write apt data
    print("Writing apt data, assuming the keys and train file are ordered the same")
    write_apt_data(train_keys_file, train_cdtm_file, train_out_file, author_hc)
    write_apt_data(holdout_keys_file, holdout_cdtm_file, holdout_out_file, author_hc)


if __name__ == "__main__":
    main()
