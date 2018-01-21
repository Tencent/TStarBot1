#!/usr/bin/env python
import sys
import hashlib
import time


seed_str = str(time.time())
line_id = 0
for line in sys.stdin:
    line = line.strip()
    key = hashlib.sha224(line + seed_str + str(line_id)).hexdigest()
    line_id += 1
    print("%s\t%s" % (key, line))
