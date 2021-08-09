import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn, VecEnvObs, VecEnv


class ConfidenceWrapper(VecEnvWrapper):

    def __init__(self, venv: VecEnv, confidence: float):
        super().__init__(venv)
        self.confidence = confidence

    def reset(self) -> VecEnvObs:
        return self.venv.reset()

    def step_async(self, actions: np.ndarray) -> None:
        new_act = actions.copy()
        new_act = new_act * self.confidence
        self.venv.step_async(np.clip(new_act, -1, 1))

    def step_wait(self) -> VecEnvStepReturn:
        return self.venv.step_wait()
