from stable_baselines3 import PPO
import os, sys
import pathlib
from rlgym.utils.action_parsers.discrete_act import DiscreteAction

class Agent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
            "device": "cpu"
        }
        self.actor = PPO.load(str(_path) + '/carballai_500m_model', custom_objects=custom_objects)
        self.parser = DiscreteAction()

    def act(self, state):
        action = self.actor.predict(state, deterministic=True)
        x = self.parser.parse_actions(action[0], state)
        return x[0]