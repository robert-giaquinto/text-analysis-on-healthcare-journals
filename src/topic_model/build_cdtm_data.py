from __future__ import division, print_function, absolute_import
import os


def day_counts(keys_file):
    rval = {}
    with open(keys_file, "r") as f:
        for line in f:
            fields = line.split("\t")
            day = int(round(float(fields[3])))
            if day in rval:
                rval[day] += 1
            else:
                rval[day] = 1
    return rval


def build_data(keys_file, out_file, ldac_file):
    days = day_counts(keys_file)
    n_days = len(days)
    print("Found", n_days, "unique timestamps")
    print("Writing day ", end='')
    
    with open(out_file, 'wb') as out, open(ldac_file, "r") as ldac:
        out.write(str(n_days) + '\n')
        for day, n_docs in sorted(days.iteritems()):
            print(day, end=', ')
            out.write(str(day) + '\n')
            out.write(str(n_docs) + '\n')
            for i in range(n_docs):
                bow = ldac.readline()
                out.write(bow)
    print('\nDone!')

    
def main():
    data_dir = '/home/srivbane/shared/caringbridge/data/cdtm/'
    keys_file = 'train_10k_sites_known_hc_key.txt'
    ldac_file = 'train-mult.dat'
    out_file = 'train.dat'
    build_data(data_dir + keys_file, data_dir + out_file, data_dir + ldac_file)

    keys_file = 'test_10k_sites_known_hc_key.txt'
    ldac_file = 'holdout-mult.dat'
    out_file = 'holdout.dat'
    build_data(data_dir + keys_file, data_dir + out_file, data_dir + ldac_file)

    

if __name__ == "__main__":
    main()
