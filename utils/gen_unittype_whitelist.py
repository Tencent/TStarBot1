import os
import sys
import random
import cPickle as pickle
import StringIO
import tarfile
import gzip
from collections import Counter


def count_unittype(filepath):
    """Format tar_of_gzip supports fast random access."""
    with tarfile.open(filepath, 'r') as tar:
        all_contents = []
        counter = Counter()
        for tarinfo in tar.getmembers():
            f = tar.extractfile(tarinfo)
            content = f.read()
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            frame = pickle.loads(gfile.read())
            counter += Counter(frame["observation"]["screen"][6].flatten())
    return counter

                
root_dir = sys.argv[1]
num_sampled_files = int(sys.argv[2])
counter = Counter()
for filename in random.sample(os.listdir(root_dir), num_sampled_files):
    if filename.endswith(".tar"):
        filepath = os.path.join(root_dir, filename)
        counter += count_unittype(filepath)
        print(len(counter))
        print(sorted(counter.keys()))
