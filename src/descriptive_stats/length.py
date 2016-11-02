import json
import os
import unicodedata
import numpy as np
#import matplotlib.pyplot as plt
#from numpy import *
import time
file_name = '/home/srivbane/shared/caringbridge/data/raw/journal.json'
len_list = []
start = time.time()
all = 0
with_bd = 0
with open(file_name,'r') as f:
    for line in f:
        js = json.loads(line)
        all+=1
        if "body" in js:
            with_bd+=1
            body = str(js["body"].encode('utf-8'))
            len_list.append(len(body))
end = time.time()
print "standard deviation: "+str(np.std(len_list))
print "mean: "+str(np.mean(len_list))
print "Time to generate length distribution: ", end-start
print "all: ",all
print "with_bd: ",with_bd



