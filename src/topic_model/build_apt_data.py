from __future__ import print_function, division, absolute_import
import os
import argparse

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
    parser = argparse.ArgumentParser(description='build data for APT model.')
    parser.add_argument('--data_dir', type=str, help='Data directory where input and output files should be/go.')
    parser.add_argument('--train_cdtm', type=str, help='train.')
    parser.add_argument('--test_cdtm', type=str, help='test.')
    parser.add_argument('--train_keys', type=str, help='train keys.')
    parser.add_argument('--test_keys', type=str, help='test keys.')
    parser.add_argument('--hc_file', type=str, help='health conditions file.')
    args = parser.parse_args()

    print('parse_sentences.py')
    print(args)

    data_dir = "/home/srivbane/shared/caringbridge/data/apt/"
    hc_file = os.path.join(args.data_dir, args.hc_file)
    hc_out_keys = os.path.join(args.data_dir, "health_condition_key.txt")
    train_keys = os.path.join(args.data_dir, args.train_keys)
    train_cdtm = os.path.join(args.data_dir, args.train_cdtm)
    train_out = os.path.join(args.data_dir, "train_apt.dat")
    
    test_keys = os.path.join(args.data_dir, args.test_keys)
    test_cdtm = os.path.join(args.data_dir, args.test_cdtm)
    test_out = os.path.join(args.data_dir, "test_apt.dat")

    # process health conditions
    print("Gathering author health condition lookup table")
    author_hc = gather_health_conditions(hc_file, hc_out_keys)

    # write apt data
    print("Writing apt data, assuming the keys and train file are ordered the same")
    write_apt_data(train_keys, train_cdtm, train_out, author_hc)
    write_apt_data(test_keys, test_cdtm, test_out, author_hc)


if __name__ == "__main__":
    main()
