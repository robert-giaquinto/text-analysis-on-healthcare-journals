from __future__ import division, print_function
import json
import argparse
import time
import types
import os

"""
This file is responsible for parsing the site_scrubbed.json file. It will
generate a file that contains the site ID, the category of the health
condition, and the name of the health condition
"""
class SiteParser(object):

    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def should_skip(self, json_dict):
        if 'healthCondition' not in json_dict: 
            return True

        healthCondition = json_dict['healthCondition']
        if 'category' not in healthCondition or not healthCondition['category'].strip(): 
            return True
        if 'name' not in healthCondition or not healthCondition['name'].strip(): 
            return True

        if 'numJournals' not in json_dict:
            return True
        if json_dict['numJournals'] == 0:
            return True
        return False

    def parse(self):
        num_parsed = 0
        num_thrown_out = 0
        input_path = self.input_path
        output_path = os.path.join(self.output_path, "parsed_site_data.txt")
        with open(input_path, 'r') as fin, open(output_path, 'wb') as fout:
            for line in fin:
                num_parsed = num_parsed + 1
                json_dict = json.loads(line)
                if self.should_skip(json_dict):
                    num_thrown_out = num_thrown_out + 1
                    continue

                site_id = json_dict['_id']
                healthCondition = json_dict['healthCondition']
                category = healthCondition['category'].encode('utf-8').strip()
                name = healthCondition['name'].encode('utf-8').strip()

                output = str(site_id) + '\t' + category + '\t' + name + '\n'
                fout.write(output)

        print("Successfully parsed site file")
        print("Percentage of population parsed: " + str(num_parsed / (num_parsed + num_thrown_out)))

def main():
    parser = argparse.ArgumentParser(description='Main program for calling multiple workers to parse the journals.json file.')
    parser.add_argument("--i", "--input_file", type=str, dest="input_file", help="Name of input site file to parse")
    parser.add_argument("--o", "--output_dir", type=str, dest="output_dir", help="Name of output directory to store generated file in")
    parser.set_defaults(verbose=True)
    args = parser.parse_args()

    parser = SiteParser(input_path = args.input_file, output_path = args.output_dir)
    start = time.time()
    parser.parse()
    end = time.time()
    print("Time to parse the file:", end - start, "seconds.")


if __name__ == "__main__":
    main()














