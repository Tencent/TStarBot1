import numpy as np
import gym


class PySC2ActionSpace(gym.Space):

    def __init__(self, action_spec_fn):
        self._functions = list(action_spec_fn().functions)

    def sample(self, available_ids):
        function_id = np.random.choice(available_ids)
        function_args = [[np.random.randint(0, size) for size in arg.sizes]
                          for arg in self._functions[function_id].args]
        return (function_id, function_args)

    def contains(self, x, available_ids):
        function_id, function_args = x
        if function_id not in available_ids:
            return False
        args_spec = self._functions[function_id].args
        if len(function_args) != len(args_spec):
            return False
        for arg, spec in zip(function_args, args_spec):
            if all([a >= b for a, b in zip(arg, spec.sizes)]):
                return False
        return True

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

    @property
    def space_attr(self):
        return self._functions


class PySC2ObservationSpace(gym.Space):

    def __init__(self, observation_spec_fn):
        self._feature_layers = observation_spec_fn()

    def sample(self, available_ids):
        raise NotImplementedError

    def contains(self, x, available_ids):
        raise NotImplementedError

    def to_jsonable(self, sample_n):
        raise NotImplementedError

    def from_jsonable(self, sample_n):
        raise NotImplementedError

    @property
    def space_attr(self):
        return self._feature_layers
