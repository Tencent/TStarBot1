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

def parse_tar_of_gzip(filepath, frame_id):
    """Format tar_of_gzip supports fast random access."""
    with tarfile.open(filepath, 'r') as tar:
        tarinfo = tar.getmembers()[frame_id]
        f = tar.extractfile(tarinfo)
        content = f.read()
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

parse_tar_of_gzip('tmp/0000e057beefc9b1e9da959ed921b24b9f0a31c63fedb8d94a1db78b58cf92c5.SC2Replay-1-2608.tar', 999)
