import time
import queue
import threading
from gym import spaces
from absl import logging


def add_input(action_queue, n):
    while True:
        if action_queue.empty():
            cmds = input("Input Action ID: ")
            if not cmds.isdigit():
                print("Input should be an interger. Skipped.")
                continue
            action = int(cmds)
            if action >=0 and action < n:
                action_queue.put(action)
            else:
                print("Invalid action. Skipped.")


class KeyboardAgent(object):
    """A random agent for starcraft."""
    def __init__(self, action_space):
        super(KeyboardAgent, self).__init__()
        logging.set_verbosity(logging.ERROR)
        assert isinstance(action_space, spaces.Discrete)
        self._action_queue = queue.Queue()
        self._cmd_thread = threading.Thread(
            target=add_input, args=(self._action_queue, action_space.n))
        self._cmd_thread.daemon = True
        self._cmd_thread.start()

    def act(self, observation, eps=0):
        #time.sleep(0.02)
        if not self._action_queue.empty():
            return self._action_queue.get()
        else:
            return 0
