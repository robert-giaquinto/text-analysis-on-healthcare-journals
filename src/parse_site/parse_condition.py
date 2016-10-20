from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

input_file = "parsed_site_scrubbed.txt"
categories = {}
cancer = {}
nlen = 0
cancer_len = 0
with open(input_file, 'r') as fin:
    for line in fin:
        nlen = nlen + 1
        split = line.split('\t')
        category = split[1]
        name = split[2]
        categories[category] = categories.get(category, 0) + 1
        if category == "Cancer":
            cancer_len = cancer_len + 1
            cancer[name] = cancer.get(name, 0) + 1  
        
labels = categories.keys()
sizes = [(size / nlen) * 100 for size in categories.values()] 
plt.pie(sizes, labels=labels, autopct="%1.1f%%")
plt.axis('equal')
plt.savefig('conditions.png')
plt.clf()

can_labels = cancer.keys()
can_sizes = [(size / cancer_len) * 100 for size in cancer.values()]
for size in can_sizes:
    print(size)
width = 1/len(can_labels)
xlocations = np.array(range(len(can_labels))) + 0.5
plt.bar(xlocations, can_sizes, width=width) 
plt.yticks(np.arange(0, 1, 0.1))
plt.ylim(ymin=0, ymax=1)
plt.xticks(xlocations + width / 2, can_labels, rotation='vertical')
plt.gca().get_xaxis().tick_bottom()
plt.gca().get_yaxis().tick_left()
plt.savefig('conditions_cancer.png')



