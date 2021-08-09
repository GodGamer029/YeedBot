import sys
print(sys.version)
import rlgym
import time

from stable_baselines3 import PPO

import numpy as np

from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils import math
from common_stuff import SimpleNet, CustomReward

env = rlgym.make(self_play=False, reward_fn=CombinedReward(reward_functions=(CustomReward(), EventReward(team_goal=50, concede=-5)), reward_weights=[float(1), float(1)]))

#Initialize PPO from stable_baselines3
model = PPO("MlpPolicy", env=env, verbose=1, learning_rate=0.0005, tensorboard_log="./tensorboard/")

#Train our agent!
model.learn(total_timesteps=int(1e6))

while True:
    obs = env.reset()
    done = False
    steps = 0
    ep_reward = 0
    t0 = time.time()
    while not done:
        action, _state = model.predict(obs, deterministic=True)
        new_obs, reward, done, state = env.step(action)
        ep_reward += reward
        obs = new_obs
        steps += 1

    length = time.time() - t0
    print("Step time: {:1.5f} | Episode time: {:.2f} | Episode Reward: {:.2f}".format(length / steps, length, ep_reward))