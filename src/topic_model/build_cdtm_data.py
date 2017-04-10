from __future__ import division, print_function, absolute_import
import os
import argparse


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
    parser = argparse.ArgumentParser(description='build cdtm data')
    parser.add_argument('--data_dir', type=str, help='Data directory where input and output files should be/go.')
    parser.add_argument('--train_keys', type=str, help='train keys file.')
    parser.add_argument('--test_keys', type=str, help='test keys file.')
    parser.add_argument('--train_out', type=str, help='train out file.')
    parser.add_argument('--test_out', type=str, help='test out file.')
    args = parser.parse_args()

    print('build_cdtm_data.py')
    print(args)

    train_keys = os.path.join(args.data_dir, args.train_keys)
    test_keys = os.path.join(args.data_dir, args.test_keys)
    train_out = os.path.join(args.data_dir, args.train_out)
    test_out = os.path.join(args.data_dir, args.test_out)
    train_ldac = os.path.join(args.data_dir, 'train-mult.dat')
    test_ldac = os.path.join(args.data_dir, 'test-mult.dat')

    build_data(train_keys, train_out, train_ldac)
    build_data(test_keys, test_out, test_ldac)
    

if __name__ == "__main__":
    main()
