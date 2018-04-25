import sys
import cPickle as pickle
import StringIO
import tarfile
import base64
import gzip

def parse_txt_of_gzip(filepath):
    """Format tar_of_gzip supports fast random access."""
    contents = []
    with open(filepath, 'r') as f:
        for line in f:
            content = base64.b64decode(line)
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            data = pickle.loads(gfile.read())
            contents.append(data)
    return contents

def  check_action_validity(frames):
    count = 0
    for i in xrange(1, len(frames) - 1):
        available = frames[i]["observation"]["available_actions"]
        for action in frames[i]["actions"]:
            if not action.function in available:
                count += 1
    print("Offset 0: invalid action count: %d" % count)

    count = 0
    for i in xrange(1, len(frames) - 1):
        available = frames[i - 1]["observation"]["available_actions"]
        for action in frames[i]["actions"]:
            if not action.function in available:
                count += 1
    print("Offset -1: invalid action count: %d" % count)

    count = 0
    for i in xrange(1, len(frames) - 1):
        available = frames[i + 1]["observation"]["available_actions"]
        for action in frames[i]["actions"]:
            if not action.function in available:
                count += 1
    print("Offset 1: invalid action count: %d" % count)

if __name__ == "__main__":
    filepath = sys.argv[1]
    data = parse_txt_of_gzip(filepath)
    check_action_validity(data)
