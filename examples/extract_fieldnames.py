import json
import os

data_dir = "/home/srivbane/shared/caringbridge/data/dev/"
files = os.listdir(data_dir)

def read_json(filename):
    with open(filename, 'r') as f:
        json_arr = f.readlines()
        json_arr = [j.replace("\n", "") for j in json_arr]
    return json_arr

for json_file in files:
    str_arr = read_json(data_dir + json_file)
    json_dict = [json.loads(s) for s in str_arr]
    fields = [j.keys() for j in json_dict]
    flat_keys = [i for f in fields for i in f]
    uniq_keys = set(flat_keys)
    for k in uniq_keys:
        print json_file, ",", k
