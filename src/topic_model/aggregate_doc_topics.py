from __future__ import division, print_function, absolute_import
import os
import subprocess
import numpy as np
import argparse

def avg_by_site(infile, outfile):
    """
    Simplest approach, just average the topic vectors for each document in a site
    
    Note, this gives equal weight to short journals and long journals
    """
    # sort the infile by site id
    check_sorted_cmd = """/bin/bash -c "sort -nc %s -t $',' -k1,1 -S %s" """ % (infile, "80%")
    try:
        subprocess.check_call(check_sorted_cmd, shell=True)
        print("File aleady sorted properly")
    except subprocess.CalledProcessError as e:
        # file isn't already sorted
        print("Sorting file.")
        cmd = """/bin/bash -c "sort -n %s -t $',' -k1,1 -o %s -S %s -T /home/srivbane/shared/caringbridge/data/tmp" """ % (infile, infile, "80%")
        subprocess.call(cmd, shell=True)

    print("Averaging doc topics per site...")
    current_site = ''
    current_topics = []
    with open(infile, 'r') as fin, open(outfile, 'wb') as fout:
        for i, line in enumerate(fin):
            if line == '\n':
                break
            if i % 1000000 == 0:
                print(i)
            
            fields = line.replace('\n', '').split(',')
            keys = fields[0:4]
            topics = [float(t) for t in fields[4:]]

            if i == 0:
                current_topics.append(topics)
                current_site = keys[0]
                continue
            
            if current_site == keys[0]:
                # more data from same site, just accumulate the data for now
                current_topics.append(topics)
            else:
                # found data from a new site
                # 1. aggregate current collection of results
                #    numpy should be faster as long as num_topics > 8-ish
                if len(current_topics) > 1:
                    topic_avg = np.array(current_topics).mean(axis=0)
                else:
                    topic_avg = current_topics[0]
                
                # 2. save current results
                fout.write(keys[0] + ',' + ','.join([str(t) for t in topic_avg]) + '\n')
                
                # 3. update variables
                current_topics = [topics]
                current_site = keys[0]
                

def concatenate_journals():
    """
    Better approach:
    concatenate the terms from each journal for each site.
    instead of giving each journal equal weight, this will give each word equal weight
    """
    pass
        
                

def main():
    parser = argparse.ArgumentParser(description='Aggregate document topic probability matrix to give topic vectors for each site.')
    parser.add_argument('-i', '--infile', type=str, help='File containing document topic probabilities.')
    parser.add_argument('-o', '--outfile', type=str, help='Where to write the output of a topic probabilities for each site.')
    parser.add_argument('--log', dest="verbose", action="store_true", help='Add this flag to have progress printed to log.')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    print('aggregate_doc_topics.py')
    print(args)
    
    avg_by_site(args.infile, args.outfile)



if __name__ == "__main__":
    main()
