from typing import Tuple

import numpy as np

from stable_baselines3 import PPO

from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy

from rlgym.envs import Match
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards import EventReward
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, GoalScoredCondition
from rlgym.utils.state_setters import DefaultState

from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from rlgym_tools.sb3_utils.sb3_multidiscrete_wrapper import SB3MultiDiscreteWrapper

from superiorobs import SuperiorObs
from common_stuff import SimpleNet, CustomReward, SkipReward, Mish
from hardcoded_bot_wrapper import HardcodedBotWrapper
import torch
from cool_bot_logics import supersonicml_atba
from confidence_wrapper import ConfidenceWrapper


def main():
    frame_skip = 8  # Number of ticks to repeat an action
    half_life_seconds = 7  # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))  # Quick mafs
    horizon = 5 * round(1 / (1 - gamma))  # Inspired by OpenAI Five

    print(f"fps={fps}, gamma={gamma}, horizon={horizon}")

    def get_match(num_match, num_psyonix=0, num_hardcoded=0) -> Tuple[Match, np.ndarray]:
        is_psyonix = num_match < num_psyonix
        is_hardcoded = not is_psyonix and num_match < (num_hardcoded + num_psyonix)
        playing_arr = np.array([0], dtype=int)
        if not is_psyonix:
            if is_hardcoded:
                playing_arr = np.append(playing_arr, [True])
            else:  # self play
                playing_arr = np.append(playing_arr, [False])

        return Match(
            team_size=1,
            tick_skip=frame_skip,
            spawn_opponents=is_psyonix,
            reward_function=CombinedReward(reward_functions=(CustomReward(), EventReward(team_goal=5, concede=-3)),
                                           reward_weights=[float(1), float(1)]),  # Simple reward since example code
            self_play=(not is_psyonix),
            terminal_conditions=[TimeoutCondition(round(fps * 300)), GoalScoredCondition()],  # Some basic terminals
            obs_builder=SuperiorObs(),  # very superior, best default
            state_setter=DefaultState()
        ), playing_arr

    rl_path = r"E:\Steam\steamapps\common\rocketleague\Binaries\Win64\RocketLeague.exe"  # Path to Epic installation

    num_instances = 7
    num_psyonix_bots = 0
    num_hardcoded_bots = 2
    match_configs = [get_match(i, num_psyonix=num_psyonix_bots, num_hardcoded=num_hardcoded_bots) for i in
                     range(num_instances)]
    matches = [match_configs[i][0] for i in range(num_instances)]
    hardcoded_mask = np.array([], dtype=int)
    for i in range(num_instances):
        hardcoded_mask = np.append(hardcoded_mask, match_configs[i][1])

    env = SB3MultipleInstanceEnv(rl_path, matches, num_instances,
                                 wait_time=25)  # Start 2 instances, waiting 60 seconds between each
    env = VecCheckNan(env)  # Optional
    env = VecMonitor(env)  # Recommended, logs mean reward and ep_len to Tensorboard
    env = HardcodedBotWrapper(env, hardcoded_mask=hardcoded_mask, bot_logic=supersonicml_atba)
    env = ConfidenceWrapper(env, confidence=1.1)
    # env = SB3MultiDiscreteWrapper(env)  # Convert action space to multidiscrete
    # env = VecNormalize(env, norm_obs=False, gamma=gamma)    # Highly recommended, normalizes rewards

    # Hyperparameters presumably better than default; inspired by original PPO paper
    model = PPO(
        MlpPolicy,
        env,
        n_epochs=10,  # PPO calls for multiple epochs, SB3 does early stopping to maintain target kl
        target_kl=0.02 / 1.5,  # KL to aim for (divided by 1.5 because it's multiplied later for unknown reasons)
        learning_rate=5e-5,  # Around this is fairly common for PPO
        ent_coef=0.01,  # From PPO Atari
        vf_coef=1.,  # From PPO Atari
        gamma=gamma,  # Gamma as calculated using half-life
        verbose=3,  # Print out all the info as we're going
        batch_size=horizon,  # Batch size as high as possible within reason
        n_steps=horizon,  # Number of steps to perform before optimizing network
        tensorboard_log="out/logs",  # `tensorboard --logdir out/logs` in terminal to see graphs
        device="cpu",  # Uses GPU if available
        policy_kwargs=dict([('activation_fn', Mish)])
    )
    model = model.load(r"E:\replays\scripts\policy\rl_model_20749668_steps.zip", env=env, learning_rate=1e-5, vf_coef=1,
                       n_envs=env.num_envs, policy_kwargs=dict([('activation_fn', Mish)]), target_kl=0.02 / 1.5)
    # Quickly scramble some values so we get more jumps and dodges!
    do_scramble = False
    if do_scramble:
        pol = model.policy  # type: ActorCriticPolicy
        actionNet = pol.action_net  # type: torch.nn.Linear
        print("actionNet weights", actionNet.weight, "biases", actionNet.bias)
        print("dimensions", actionNet.weight.size(0), actionNet.weight.size(1), actionNet.bias.size(0))
        scrambleTensor = torch.tensor(
            [0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 0, 0, 0,
             0])

        with torch.no_grad():
            actionNet.bias[15] = actionNet.bias[15] + torch.rand(1) * -0.05  # jump bias
            actionNet.bias[16] = actionNet.bias[16] + torch.rand(1) * 0.15

            for i in range(actionNet.weight.size(0)):
                if scrambleTensor[i] > 0.5:
                    actionNet.weight[i] = actionNet.weight[i] * (torch.rand(actionNet.weight.size(1)) + 0.5)
                actionNet.weight[i] = actionNet.weight[i] + (
                            (torch.rand(actionNet.weight.size(1)) - 0.5) * scrambleTensor[i])

    # Save model every so often
    # Divide by num_envs (number of agents) because callback only increments every time all agents have taken a step
    callback = CheckpointCallback(round(250_000 / env.num_envs), "policy")

    # import code
    # code.interact(local=locals())
    model.learn(100_000_000, callback=callback)


if __name__ == '__main__':  # Required for multiprocessing
    main()
