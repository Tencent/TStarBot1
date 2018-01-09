#!/usr/bin/python
# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Dump out stats about all the actions that are in use in a set of replays."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle as pickle
import gzip
import base64
import json
import multiprocessing
import os
import signal
import sys
import threading
import time

from future.builtins import range    # pylint: disable=redefined-builtin
import six
from six.moves import queue

from pysc2 import run_configs
from pysc2.lib import features
from pysc2.lib import point
from pysc2.lib import protocol
from pysc2.lib import remote_controller
from pysc2.lib import gfile
from s2clientprotocol import sc2api_pb2 as sc_pb

from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS
flags.DEFINE_integer("parallel", 16, "How many instances to run in parallel.")
flags.DEFINE_integer("step_mul", 8, "How many game steps per observation.")
flags.DEFINE_integer("resolution", 64, "Resolution for observation map.")
flags.DEFINE_string("replays", None, "Path to a directory of replays.")
flags.DEFINE_string("output_dir", None, "Path to a write replay data to.")
flags.mark_flag_as_required("replays")
flags.mark_flag_as_required("output_dir")


def valid_replay(info, ping):
    """Make sure the replay isn't corrupt, and is worth looking at."""
    if (info.HasField("error") or
            info.base_build != ping.base_build or    # different game version
            info.game_duration_loops < 1000 or
            len(info.player_info) != 2):
        # Probably corrupt, or just not interesting.
        return False
    for p in info.player_info:
        if p.player_apm < 10 or p.player_mmr < 1000:
            # Low APM = player just standing around.
            # Low MMR = corrupt replay or player who is weak.
            return False
    return True


class ReplayProcessor(multiprocessing.Process):
    """A Process that pulls replays and processes them."""

    def __init__(self, proc_id, run_config, replay_queue, output_dir):
        super(ReplayProcessor, self).__init__()
        self.run_config = run_config
        self.replay_queue = replay_queue
        self._output_dir = output_dir
        size = point.Point(FLAGS.resolution, FLAGS.resolution)
        self.interface = sc_pb.InterfaceOptions(
                raw=True, score=False,
                feature_layer=sc_pb.SpatialCameraSetup(width=24))
        size.assign_to(self.interface.feature_layer.resolution)
        size.assign_to(self.interface.feature_layer.minimap_resolution)

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())
        while True:
            print("Starting up a new SC2 instance.")
            try:
                with self.run_config.start() as controller:
                    ping = controller.ping()
                    for _ in range(300):
                        try:
                            replay_path = self.replay_queue.get()
                        except queue.Empty:
                            return
                        try:
                            replay_name = os.path.basename(replay_path)
                            replay_data = self.run_config.replay_data(
                                replay_path)
                            info = controller.replay_info(replay_data)
                            if valid_replay(info, ping):
                                map_data = None
                                if info.local_map_path:
                                    map_data = self.run_config.map_data(
                                        info.local_map_path)
                                for player_id in [1, 2]:
                                    print("Starting %s from player %s's "
                                          "perspective" % (replay_name[:10],
                                                           player_id))
                                    self.process_replay(controller,
                                                        replay_data,
                                                        map_data,
                                                        player_id,
                                                        replay_name)
                            else:
                                print("Replay %s is invalid." % replay_name[:10])
                        finally:
                            self.replay_queue.task_done()
            except (protocol.ConnectionError, protocol.ProtocolError,
                    remote_controller.RequestError):
                print("Replay crashed.")
            except KeyboardInterrupt:
                return

    def process_replay(self, controller, replay_data, map_data, player_id,
                       replay_name):
        """Process a single replay, write to a gzip file."""
        controller.start_replay(sc_pb.RequestStartReplay(
                replay_data=replay_data,
                map_data=map_data,
                options=self.interface,
                observed_player_id=player_id))
        feat = features.Features(controller.game_info())
        controller.step()
        all_frames = []
        while True:
            cur_frame = {}
            obs = controller.observe()
            cur_frame['observation'] = feat.transform_obs(obs.observation)
            cur_frame['actions'] = []
            for action in obs.actions:
                try:
                    cur_frame['actions'].append(feat.reverse_action(action))
                except ValueError:
                    pass
            all_frames.append(cur_frame)
            if obs.player_result:
                assert obs.player_result[player_id - 1].player_id == player_id
                result = obs.player_result[player_id - 1].result
                break
            controller.step(FLAGS.step_mul)
        for i in range(len(all_frames)):
            all_frames[i]['result'] = result
            all_frames[i]['replay_name'] = replay_name
            all_frames[i]['player_id'] = player_id
        self._save_gzip(all_frames, "%s-%d.gz" % (replay_name[:10], player_id))

    def _save_gzip(self, all_frames, filename):
        filepath = os.path.join(self._output_dir, filename)
        with gzip.open(filepath, 'w') as file:
            for frame in all_frames:
                file.write(base64.b64encode(pickle.dumps(frame)) + '\n')
        print("Replay data saved in %s." % filepath)


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
    logging.set_verbosity(logging.ERROR)
    run_config = run_configs.get()

    if not gfile.Exists(FLAGS.replays):
        sys.exit("{} doesn't exist.".format(FLAGS.replays))

    try:
        # For some reason buffering everything into a JoinableQueue makes the
        # program not exit, so save it into a list then slowly fill it into the
        # queue in a separate thread. Grab the list synchronously so we know there
        # is work in the queue before the SC2 processes actually run, otherwise
        # The replay_queue.join below succeeds without doing any work, and exits.
        print("Getting replay list:", FLAGS.replays)
        replay_list = sorted(run_config.replay_paths(FLAGS.replays))
        print(len(replay_list), "replays found.\n")
        replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.parallel):
            p = ReplayProcessor(i, run_config, replay_queue, FLAGS.output_dir)
            p.daemon = True
            p.start()
            time.sleep(1)    # Stagger startups, otherwise conflict somehow

        replay_queue.join()    # Wait for the queue to empty.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")


if __name__ == "__main__":
        app.run(main)
