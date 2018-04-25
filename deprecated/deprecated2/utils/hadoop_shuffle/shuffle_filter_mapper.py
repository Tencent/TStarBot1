#!/usr/bin/env python
import sys
import hashlib
import time
import base64
import gzip
import StringIO
import cPickle as pickle
import numpy as np


sys.path.append(".")
seed_str = str(time.time())
line_id = 0
for line in sys.stdin:
    line = line.strip()
    if line_id == 0:
        content = base64.b64decode(line)
        gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
        data = pickle.loads(gfile.read())
        mean_x = np.nonzero(data['observation']['minimap'][1])[0].mean()
        if mean_x < 30:
            position = 'left_top'
        elif mean_x > 34:
            position = 'right_bottom'
        else:
            position = 'uncertain'
    if position == 'right_bottom':
        key = hashlib.sha224(line + seed_str + str(line_id)).hexdigest()
        print("%s\t%s" % (key, line))
    line_id += 1
