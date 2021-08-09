import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from collections import OrderedDict
from rlgym.utils.gamestates import GameState, PlayerData
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn, VecEnvWrapper
from typing import Optional, Tuple

@torch.jit.script
def mish(x):
    return x * (torch.tanh(F.softplus(x)))

class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))

normalizer = np.array([1/4100, 1/6000, 1/2000, 1/4000, 1/4000, 1/4000, 1/6, 1/6, 1/6, 1/4100, 1/6000, 1/2000, 1/2300, 1/2300, 1/2300, 1/5.5, 1/5.5, 1/5.5, 1, 1, 1, 1, 1, 1, 1, 1/4100, 1/6000, 1/2000, 1/2300, 1/2300, 1/2300, 1/5.5, 1/5.5, 1/5.5, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
inputs = 41
outputs = 1
class SimpleNet(nn.Module):

    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(inputs, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, outputs)

    def forward(self, x):
        x = mish(self.fc1(x))
        x = mish(self.fc2(x))
        x = mish(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

    def loadFromFile(self, path):
        a = np.fromfile(path, dtype=np.float32)

        w1 = np.empty((64, inputs))
        b1 = np.empty((w1.shape[0]))
        w2 = np.empty((64, 64))
        b2 = np.empty((w2.shape[0]))
        w3 = np.empty((32, 64))
        b3 = np.empty((w3.shape[0]))
        w4 = np.empty((outputs, 32))
        b4 = np.empty((w4.shape[0]))
        print("pred size:", w1.size + b1.size + w2.size + b2.size + w3.size + b3.size + w4.size + b4.size, "actual", a.size)
        
        cur = 0
        w1 = np.reshape(a[0:w1.size], w1.shape)
        cur = cur + w1.size
        b1 = a[cur:b1.size+cur]
        cur = cur + b1.size

        w2 = np.reshape(a[cur:w2.size+cur], w2.shape)
        cur = cur + w2.size
        b2 = a[cur:b2.size+cur]
        cur = cur + b2.size

        w3 = np.reshape(a[cur:w3.size+cur], w3.shape)
        cur = cur + w3.size
        b3 = a[cur:b3.size+cur]
        cur = cur + b3.size

        w4 = np.reshape(a[cur:w4.size+cur], w4.shape)
        cur = cur + w4.size
        b4 = a[cur:b4.size+cur]
        cur = cur + b4.size

        
        #print([(k, v.shape) for k, v in net.state_dict().items()])
        new_state_dict = OrderedDict({'fc1.weight': torch.from_numpy(w1), 'fc1.bias': torch.from_numpy(b1), 'fc2.weight': torch.from_numpy(w2), 'fc2.bias': torch.from_numpy(b2), 'fc3.weight': torch.from_numpy(w3), 'fc3.bias': torch.from_numpy(b3), 'fc4.weight': torch.from_numpy(w4), 'fc4.bias': torch.from_numpy(b4)})
        self.load_state_dict(new_state_dict, strict=True)

net = SimpleNet()
net.loadFromFile('E:\\replays\\scripts\\network.ptyang')
net.eval()
tracedNet = torch.jit.trace(net, torch.rand((inputs)))

class CustomReward(RewardFunction):
    def reset(self, initial_state: GameState):
        self.last_assessment = 0
        self.timeInStandoff = {}
        self.dodgeRewardTimeout = {}
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        with torch.no_grad():
            ball = state.ball
            bInp = np.concatenate((ball.position, ball.linear_velocity, ball.angular_velocity), axis=None)

            teams = [[], []]
            for p in state.players:
                if p.team_num == 0:
                    teams[0].append(p)
                else:
                    teams[1].append(p)

            p0 = teams[0][0].car_data
            t0_inp = np.concatenate((p0.position, p0.linear_velocity, p0.angular_velocity, p0.forward(), p0.up(), teams[0][0].boost_amount), axis=None)

            p1 = teams[1][0].car_data
            t1_inp = np.concatenate((p1.position, p1.linear_velocity, p1.angular_velocity, p1.forward(), p1.up(), teams[1][0].boost_amount), axis=None)

            inp = np.concatenate((bInp, t0_inp, t1_inp), axis=None)

            state_assessment = tracedNet(torch.from_numpy((inp * normalizer).astype('float32'))).item()
            state_assessment = state_assessment * 2 - 1
            if player.team_num == 0:
                state_assessment = -state_assessment
            newReward = state_assessment - self.last_assessment
            self.last_assessment = state_assessment

            '''
            carSpeed = min(float(2300), np.linalg.norm(player.car_data.linear_velocity))
            carSpeedThreshold = 300
            if carSpeed > carSpeedThreshold:
                carSpeedReward = 0.05 * np.sqrt((carSpeed - carSpeedThreshold) / (2300 - carSpeedThreshold))
            else:
                carSpeedReward = 0

            if not player.car_id in self.dodgeRewardTimeout:
                self.dodgeRewardTimeout[player.car_id] = 0
            if self.dodgeRewardTimeout[player.car_id] > 0:
                self.dodgeRewardTimeout[player.car_id] -= 1
            car_in_air_posfactor = 0
            car_in_air_reward = 0
            if not player.on_ground:
                if player.has_flip:
                    car_in_air_posfactor = 0.1
                    if player.ball_touched and self.dodgeRewardTimeout[player.car_id] == 0:
                        self.dodgeRewardTimeout[player.car_id] = 120 / 8
                        car_in_air_reward += 0.05
                elif (not player.has_flip) and np.linalg.norm(player.car_data.angular_velocity) > 4:
                    car_in_air_posfactor = 1
                    if player.ball_touched and self.dodgeRewardTimeout[player.car_id] == 0:
                        self.dodgeRewardTimeout[player.car_id] = 120 / 8
                        car_in_air_reward += 0.5
                else:
                    car_in_air_posfactor = 0.2
                    if player.ball_touched and self.dodgeRewardTimeout[player.car_id] == 0:
                        self.dodgeRewardTimeout[player.car_id] = 120 / 8
                        car_in_air_reward += 0.25

            car_in_air_posfactor += 1
            if newReward > 0: # increase reward
                newReward *= car_in_air_posfactor
            else: # reduce penalty
                newReward *= 1 / car_in_air_posfactor
            #car_in_air_reward *= 3

            ballSpeed = np.linalg.norm(ball.linear_velocity)
            ballSpeedThreshold = 150
            if ballSpeed < ballSpeedThreshold:
                if not player.car_id in self.timeInStandoff:
                    self.timeInStandoff[player.car_id] = 0
                self.timeInStandoff[player.car_id] += 1
                start_t = 2.5 * 120 / 8
                end_t = 10 * 120 / 8
                ballSpeedPenalty = np.interp(self.timeInStandoff[player.car_id], [start_t, end_t], [-0.001, -0.05]) * 1 - (ballSpeed / ballSpeedThreshold)
            else:
                ballSpeedPenalty = 0
                self.timeInStandoff[player.car_id] = 0
            if np.linalg.norm(ball.position - np.array([0, 0, 92])) < 30:
                ballSpeedPenalty = 0
                self.timeInStandoff[player.car_id] = 0
            '''
            newReward = newReward - 0.05  # do something! *pokes network nervously*
            # + carSpeedReward + ballSpeedPenalty + car_in_air_reward
            return newReward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return 0


class SkipReward(RewardFunction):

    def __init__(self, other: RewardFunction, skip_num: int = 2):
        super().__init__()
        self.other = other
        self.skip_num = skip_num

    def reset(self, initial_state: GameState):
        self.skip = {}
        self.other.reset(initial_state)
        pass

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if player.car_id in self.skip:
            self.skip[player.car_id] = self.skip[player.car_id] + 1
            if self.skip[player.car_id] >= self.skip_num:
                self.skip[player.car_id] = 0
                return self.other.get_reward(player, state, previous_action) * self.skip_num
        else:
            self.skip[player.car_id] = 1
        return 0
        

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.other.get_final_reward(player, state, previous_action)

