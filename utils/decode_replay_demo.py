import sys
import cPickle as pickle
import StringIO
import tarfile
import base64
import gzip

def parse_gzip(filepath):
    """Format gzip connot support random access."""
    with gzip.open(filepath, 'r') as f:
        for line in f:
            data = pickle.loads(base64.b64decode(line))
            print_data(data)

def parse_tar_of_gzip(filepath):
    """Format tar_of_gzip supports fast random access."""
    with tarfile.open(filepath, 'r') as tar:
        for tarinfo in tar.getmembers():
            f = tar.extractfile(tarinfo)
            content = f.read()
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            data = pickle.loads(gfile.read())
            print_data(data)

def parse_txt_of_gzip(filepath):
    """Format tar_of_gzip supports fast random access."""
    with open(filepath, 'r') as f:
        for line in f:
            content = base64.b64decode(line)
            gfile = gzip.GzipFile(fileobj=StringIO.StringIO(content))
            data = pickle.loads(gfile.read())
            print_data(data)

def print_data(data):
    obs = data["observation"]
    actions = data["actions"]
    replay_name = data["replay_name"]
    player_id = data["player_id"]
    result = data["result"] # 1 victory  2 defeat
    print("Observations: %s" % obs)
    print("Actions: %s" % actions)
    print("Replay Name: %s" % replay_name)
    print("Player View ID : %d" % player_id)
    print("Result (1-Victory, 2-Defeat) : %d" % result)


if __name__ == "__main__":
    filepath = sys.argv[1]
    if filepath.endswith('.frame'):
        parse_txt_of_gzip(filepath)
    elif filepath.endswith('.gz'):
        parse_gzip(filepath)
    elif filepath.endswith('.tar'):
        parse_tar_of_gzip(filepath)
    else:
        raise NotImplementedError
