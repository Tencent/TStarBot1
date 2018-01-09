import cPickle as pickle
import base64
import gzip

def parse(filepath):
    with gzip.open(filepath, 'r') as f:
        for line in f:
            data = pickle.loads(base64.b64decode(line))
            obs = data["observation"]
            actions = data["actions"]
            replay_name = data["replay_name"]
            player_id = data["player_id"]
            result = data["result"] # 1 victory  2 defeat
            print(obs)
            print(actions)
            print(replay_name)
            print(player_id)
            print(result)

parse('tmp/0002b71a92-1.gz')
