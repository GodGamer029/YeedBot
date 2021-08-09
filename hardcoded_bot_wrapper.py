import numpy as np
from rlgym.utils.gamestates import GameState
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs, VecEnv
from typing import Callable, List, Union, Any


class HardcodedBotWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, hardcoded_mask: np.ndarray, bot_logic: Union[Callable[[np.ndarray, GameState, int], np.ndarray], List[Callable[[np.ndarray, GameState, int], np.ndarray]]]):
        super().__init__(venv)
        self.hardcoded_mask = hardcoded_mask
        self.num_envs = self.num_envs - np.sum(hardcoded_mask)
        self.bot_obs = None
        self.bot_info = None
        if callable(bot_logic): # just duplicate the logic for every bot
            self.bot_logic = [bot_logic for i in range(np.sum(hardcoded_mask))]
        else:
            self.bot_logic = bot_logic
        assert len(self.bot_logic) == np.sum(hardcoded_mask)

    def update_obs(self, new_obs: np.ndarray):
        updated_obs = np.empty((new_obs.shape[0] - np.sum(self.hardcoded_mask), new_obs.shape[1]))
        self.bot_obs = np.empty((np.sum(self.hardcoded_mask), new_obs.shape[1]))
        ind = 0
        bot_ind = 0
        assert new_obs.shape[0] == self.hardcoded_mask.shape[0], (new_obs.shape[0], self.hardcoded_mask.shape[0])
        for i in range(self.hardcoded_mask.shape[0]):
            shall_remove = self.hardcoded_mask[i]
            if shall_remove == 1:
                self.bot_obs[bot_ind] = new_obs[i]
                bot_ind += 1
                continue
            updated_obs[ind] = new_obs[i]
            ind += 1
        return updated_obs

    def remove_any(self, new_any: np.ndarray):
        assert new_any.shape[0] == self.hardcoded_mask.shape[0], (new_any.shape[0], self.hardcoded_mask.shape[0])
        if new_any.ndim == 1:
            updated_rew = np.empty(new_any.shape[0] - np.sum(self.hardcoded_mask))
        else:
            updated_rew = np.empty((new_any.shape[0] - np.sum(self.hardcoded_mask), new_any.shape[1]))
        ind = 0
        for i in range(self.hardcoded_mask.shape[0]):
            shall_remove = self.hardcoded_mask[i]
            if shall_remove == 1:
                continue
            updated_rew[ind] = new_any[i]
            ind += 1
        return updated_rew

    def update_infos(self, new_any):
        assert len(new_any) == self.hardcoded_mask.shape[0], (len(new_any), self.hardcoded_mask.shape[0])
        updated = []
        self.bot_info = []
        bot_ind = 0
        for i in range(self.hardcoded_mask.shape[0]):
            shall_remove = self.hardcoded_mask[i]
            if shall_remove == 1:
                self.bot_info.append(new_any[i])
                bot_ind += 1
                continue
            updated.append(new_any[i])
        return updated

    def reset(self) -> VecEnvObs:
        return self.update_obs(self.venv.reset())

    def get_bot_action(self, bot_ind: int, car_ind: int) -> np.ndarray:
        if self.bot_info is None or len(self.bot_info) <= bot_ind or 'state' not in self.bot_info[bot_ind]:
            return np.zeros(8)
        return self.bot_logic[bot_ind](self.bot_obs[bot_ind], self.bot_info[bot_ind].get('state'), car_ind)

    def step_async(self, actions: np.ndarray) -> None:
        new_act = np.zeros((self.hardcoded_mask.shape[0], actions.shape[1]), dtype=actions.dtype)

        bot_ind = 0
        act_ind = 0
        indices_since_last_bot = 0
        # we keep a track of when the last bot was, to estimate where the last game's players end and this game's players begin
        # this breaks horribly when there is a game with only regular players in between, so you better push all those to  the back
        last_was_bot = False
        for i in range(self.hardcoded_mask.shape[0]):
            is_bot = self.hardcoded_mask[i]
            if is_bot == 1:
                new_act[i] = self.get_bot_action(bot_ind, car_ind=indices_since_last_bot)
                bot_ind += 1
                last_was_bot = True
                indices_since_last_bot += 1
                continue
            if last_was_bot:
                indices_since_last_bot = 0
                last_was_bot = False
            indices_since_last_bot += 1

            new_act[i] = actions[act_ind]
            act_ind += 1
        self.venv.step_async(new_act)

    def step_wait(self) -> VecEnvStepReturn:
        obs, rewards, dones, infos = self.venv.step_wait()
        return self.update_obs(obs), self.remove_any(rewards), self.remove_any(dones), self.update_infos(infos)
