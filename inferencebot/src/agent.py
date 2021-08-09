# import os

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch
import torch.nn.functional as F
from obs.old_superiorobs import OldSuperiorObs
import os

class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.model = PPO.load(os.path.join(cur_dir, "yeed_bot_v1.zip"), custom_objects={'n_envs': 1})
        self.obs = OldSuperiorObs()
        pass

    def act(self, state):
        # Evaluate your model here

        action = self.model.predict(state, deterministic=True)
        return action[0]
