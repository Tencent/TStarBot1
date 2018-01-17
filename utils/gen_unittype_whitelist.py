import os
import sys
import random
import cPickle as pickle
import StringIO
import base64
import gzip
from collections import Counter


def count_unittype(filepath):
    with open(filepath, 'r') as f:
        counter = Counter()
        for line in f:
            content = base64.b64decode(line)
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            frame = pickle.loads(gfile.read())
            counter += Counter(frame["observation"]["screen"][6].flatten())
    return counter

                
root_dir = sys.argv[1]
num_sampled_files = int(sys.argv[2])
counter = Counter()
for filename in random.sample(os.listdir(root_dir), num_sampled_files):
    if filename.startswith("part"):
        filepath = os.path.join(root_dir, filename)
        counter += count_unittype(filepath)
        print(len(counter))
        print(sorted(counter.keys()))
