import numpy as np
import gym
from gym.spaces.discrete import Discrete


class PySC2RawAction(gym.Space):

  def __init__(self, action_spec_fn):
    self._functions = list(action_spec_fn().functions)

  def sample(self, availables):
    function_id = np.random.choice(availables).item()
    function_args = [[np.random.randint(0, size) for size in arg.sizes]
                     for arg in self._functions[function_id].args]
    assert self.contains((function_id, function_args))
    return (function_id, function_args)

  def contains(self, x, availables):
    function_id, function_args = x
    if not function_id in availables: return False
    args_spec = self._functions[function_id].args
    if len(function_args) != len(args_spec): return False
    for arg, spec in zip(function_args, args_spec):
      if all([a >= b for a, b in zip(arg, spec.sizes)]): return False
    return True

  def to_jsonable(self, sample_n):
    raise NotImplementedError

  def from_jsonable(self, sample_n):
    raise NotImplementedError

  @property
  def space_attr(self):
    return self._functions


class PySC2RawObservation(gym.Space):

  def __init__(self, observation_spec_fn):
    self._feature_layers = observation_spec_fn()

  def sample(self):
    raise NotImplementedError

  def contains(self, x):
    raise NotImplementedError

  def to_jsonable(self, sample_n):
    raise NotImplementedError

  def from_jsonable(self, sample_n):
    raise NotImplementedError

  @property
  def space_attr(self):
    return self._feature_layers


class MaskDiscrete(Discrete):

  def sample(self, availables):
    x = np.random.choice(availables).item()
    assert self.contains(x, availables)
    return x

  def contains(self, x, availables):
    return super(MaskDiscrete, self).contains(x) and x in availables
