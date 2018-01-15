#!/usr/bin/env python
import sys

for line in sys.stdin:
    key, content = line.strip().split('\t')
    print(content)
