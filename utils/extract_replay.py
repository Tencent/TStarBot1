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
import StringIO
import tarfile
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
flags.DEFINE_string("filelist", None, "Replay filelist to extract from.")
flags.DEFINE_string("output_dir", None, "Path to a write replay data to.")
flags.DEFINE_boolean("filter_empty_action", True, "Filter frame with no action.")
flags.DEFINE_enum("format", 'txt_of_gzip',
                  ['txt_of_gzip', 'tar_of_gzip', 'gzip'], "Output format.")
flags.mark_flag_as_required("filelist")
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

    def __init__(self, proc_id, run_config, replay_queue, resolution,
                 output_dir, output_format='txt_of_gzip',
                 filter_empty_action=True):
        super(ReplayProcessor, self).__init__()
        self._run_config = run_config
        self._replay_queue = replay_queue
        self._output_dir = output_dir
        self._output_format = output_format
        self._filter_empty_action = filter_empty_action

        size = point.Point(resolution, resolution)
        self._interface = sc_pb.InterfaceOptions(
                raw=True, score=False,
                feature_layer=sc_pb.SpatialCameraSetup(width=24))
        size.assign_to(self._interface.feature_layer.resolution)
        size.assign_to(self._interface.feature_layer.minimap_resolution)

    def run(self):
        signal.signal(signal.SIGTERM, lambda a, b: sys.exit())
        while True:
            print("Starting up a new SC2 instance.")
            try:
                with self._run_config.start() as controller:
                    ping = controller.ping()
                    for _ in range(300):
                        try:
                            replay_path = self._replay_queue.get()
                        except queue.Empty:
                            return
                        try:
                            replay_name = os.path.basename(replay_path)
                            replay_data = self._run_config.replay_data(
                                replay_path)
                            info = controller.replay_info(replay_data)
                            if valid_replay(info, ping):
                                map_data = None
                                if info.local_map_path:
                                    map_data = self._run_config.map_data(
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
                            self._replay_queue.task_done()
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
                options=self._interface,
                observed_player_id=player_id))
        feat = features.Features(controller.game_info())
        controller.step()
        all_frames = []
        done = False
        while not done:
            cur_frame = {}
            obs = controller.observe()
            if obs.player_result:
                assert obs.player_result[player_id - 1].player_id == player_id
                result = obs.player_result[player_id - 1].result
                done = True
            cur_frame['actions'] = []
            for action in obs.actions:
                try:
                    cur_frame['actions'].append(feat.reverse_action(action))
                except ValueError:
                    pass
            if len(cur_frame['actions']) > 0 or not self._filter_empty_action:
                cur_frame['observation'] = feat.transform_obs(obs.observation)
                all_frames.append(cur_frame)
            if not done:
                controller.step(FLAGS.step_mul)
        for i in range(len(all_frames)):
            all_frames[i]['result'] = result
            all_frames[i]['replay_name'] = replay_name
            all_frames[i]['player_id'] = player_id

        if self._output_format == "txt_of_gzip":
            self._save_to_txt_of_gzip(
                all_frames,
                "%s-%d.frame" % (replay_name, player_id))
        elif self._output_format == "gzip":
            self._save_to_gzip(
                all_frames,
                "%s-%d-%d.gzip" % (replay_name, player_id, len(all_frames)))
        elif self._output_format == "tar_of_gzip":
            self._save_to_tar_of_gzip(
                all_frames,
                "%s-%d-%d.tar" % (replay_name, player_id, len(all_frames)))
        else:
            raise NotImplementedError

    def _save_to_gzip(self, all_frames, filename):
        filepath = os.path.join(self._output_dir, filename)
        with gzip.open(filepath, 'w') as file:
            for frame in all_frames:
                file.write(base64.b64encode(pickle.dumps(frame)) + '\n')
        print("Replay data saved in %s." % filepath)

    def _save_to_txt_of_gzip(self, all_frames, filename):
        filepath = os.path.join(self._output_dir, filename)
        with open(filepath, 'w') as file:
            for frame in all_frames:
                input_io = StringIO.StringIO()
                with gzip.GzipFile(fileobj=input_io, mode="w") as gfile:
                    gfile.write(pickle.dumps(frame))
                file.write(base64.b64encode(input_io.getvalue()) + '\n')
        print("Replay data saved in %s." % filepath)

    def _save_to_tar_of_gzip(self, all_frames, filename):
        filepath = os.path.join(self._output_dir, filename)
        with tarfile.open(filepath, 'w') as tar:
            for i, frame in enumerate(all_frames):
                input_io = StringIO.StringIO()
                with gzip.GzipFile(fileobj=input_io, mode="w") as gfile:
                    gfile.write(pickle.dumps(frame))
                compressed_io = StringIO.StringIO(input_io.getvalue())
                info = tarfile.TarInfo(name=str(i))
                info.size = len(compressed_io.buf)
                tar.addfile(tarinfo=info, fileobj=compressed_io)
        print("Replay data saved in %s." % filepath)


def replay_queue_filler(replay_queue, replay_list):
    """A thread that fills the replay_queue with replay filenames."""
    for replay_path in replay_list:
        replay_queue.put(replay_path)


def main(unused_argv):
    """Dump stats about all the actions that are in use in a set of replays."""
    logging.set_verbosity(logging.ERROR)
    run_config = run_configs.get()

    try:
        # For some reason buffering everything into a JoinableQueue makes the
        # program not exit, so save it into a list then slowly fill it into the
        # queue in a separate thread. Grab the list synchronously so we know there
        # is work in the queue before the SC2 processes actually run, otherwise
        # The replay_queue.join below succeeds without doing any work, and exits.
        replay_list = [line.rstrip() for line in open(FLAGS.filelist)]
        print(len(replay_list), "replays found.\n")
        replay_queue = multiprocessing.JoinableQueue(FLAGS.parallel * 10)
        replay_queue_thread = threading.Thread(target=replay_queue_filler,
                                               args=(replay_queue, replay_list))
        replay_queue_thread.daemon = True
        replay_queue_thread.start()

        for i in range(FLAGS.parallel):
            p = ReplayProcessor(i, run_config, replay_queue, FLAGS.resolution,
                                FLAGS.output_dir, FLAGS.format,
                                FLAGS.filter_empty_action)
            p.daemon = True
            p.start()
            time.sleep(1)    # Stagger startups, otherwise conflict somehow

        replay_queue.join()    # Wait for the queue to empty.
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, exiting.")


if __name__ == "__main__":
    app.run(main)
