import json
import os
import unicodedata
import collections
import time
from collections import defaultdict

file_name = '/home/srivbane/shared/caringbridge/data/raw/journal.json'
plat_dict = dict()
start = time.time()
count = 0
with open(file_name,'r') as f:
      for line in f:
            count+=1
            js = json.loads(line)
            if "platform" in js:
                 if js["platform"] in plat_dict:
                     plat_dict[str(js["platform"])]+=1
                 else:
                     plat_dict[str(js["platform"])]=0
undefined_platform = count - plat_dict['iphone']-plat_dict['mobile']-plat_dict['android']
plat_dict['undefined_platform']= undefined_platform
end = time.time()
print plat_dict
print "Time to generate the platform dictionary: ",end-start
      
