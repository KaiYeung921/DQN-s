# -*- coding: utf-8 -*-
# training/sb3_train.py
import numpy as np
import gymnasium
from gymnasium import spaces as gym_spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env.hexxed import hexxed
from config import ENV_CONFIG, TRAIN_CONFIG
from tracking.mlflow_logger import log_episode as mlflow_log_episode
from training.train import ProgressTracker


class HexxedWrapper(gymnasium.Env):
    """
    Wraps hexxed (old gym API) as a gymnasium.Env for SB3 2.x compatibility.
    Calls ready() at init time so SB3 can read spaces immediately.
    Rebuilds spaces using gymnasium.spaces (not the old gym spaces).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()
        self._env = hexxed()
        self._env.ready(**ENV_CONFIG)

        nv        = ENV_CONFIG["num_vertices"]
        grid_size = nv * 2 * nv

        # gymnasium spaces — functionally identical to the old gym spaces in hexxed
        self.observation_space = gym_spaces.Box(
            low=0.0, high=1.0, shape=(grid_size,), dtype=np.float32
        )
        self.action_space = gym_spaces.Discrete(nv + 1)

    def reset(self, seed=None, options=None):
        # gymnasium reset returns (obs, info)
        obs = self._env.reset().astype(np.float32)
        return obs, {}

    def step(self, action):
        # gymnasium step returns (obs, reward, terminated, truncated, info)
        obs, reward, done, info = self._env.step(action)
        return obs.astype(np.float32), float(reward), bool(done), False, info

    def render(self):
        return self._env.render()

    # expose env internals the callback needs
    @property
    def curr_wave(self):    return self._env.curr_wave
    @property
    def wave_reward(self):  return self._env.wave_reward
    @property
    def max_reward(self):   return self._env.max_reward
    @property
    def pattern_list(self): return self._env.pattern_list


class HexxedProgressCallback(BaseCallback):
    """
    Fires every step. Mirrors the ProgressTracker logic in train.py exactly.
    Returns False (stops SB3 training) once all levels are cleared.
    """

    def __init__(self, tracker, total_steps, verbose=0):
        super().__init__(verbose)
        self.tracker     = tracker
        self.total_steps = total_steps
        self._ep_reward  = 0.0
        self._ep_num     = 0

    def _on_step(self):
        self._ep_reward += float(self.locals["rewards"][0])

        if self.locals["dones"][0]:
            env   = self.training_env.envs[0]
            level = env.curr_wave

            self.tracker.log_episode(
                self.num_timesteps,
                self._ep_reward,
                level,
                env,
            )
            mlflow_log_episode(
                self._ep_num,
                self._ep_reward,
                self.model.exploration_rate,
                level,
                loss=None,
            )

            self._ep_reward = 0.0
            self._ep_num   += 1

            if ENV_CONFIG["levels"] in self.tracker.level_clear_steps:
                return False

        return True


def train_sb3_dqn(lr, gamma, batch_size, buffer_size, target_update, hidden_dim,
                  total_steps=None):
    """
    SB3 DQN equivalent of train_dqn. Same hyperparameter signature.
    Returns (learning_speed_score, tracker) — identical shape to train_dqn.

    Uses search_timesteps from TRAIN_CONFIG by default so Optuna trials stay
    fast; pass total_steps=TRAIN_CONFIG['total_timesteps'] for a full run.

    Key difference from PyTorch DQN: SB3 uses linear epsilon decay while
    ours is multiplicative — same start/end values, slightly different curve.
    exploration_fraction=0.5 means decay finishes at the halfway point,
    giving the second half of the budget for exploitation.
    """
    if total_steps is None:
        total_steps = TRAIN_CONFIG["search_timesteps"]

    tracker = ProgressTracker(rolling_window=TRAIN_CONFIG["rolling_window"])

    env = DummyVecEnv([lambda: HexxedWrapper()])

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=lr,
        gamma=gamma,
        batch_size=batch_size,
        buffer_size=buffer_size,
        target_update_interval=target_update,
        # single hidden layer — matches DQNNetwork(obs -> hidden -> actions)
        policy_kwargs=dict(net_arch=[hidden_dim]),
        # train every step to match our every-step gradient update
        train_freq=1,
        gradient_steps=1,
        # linear decay — same endpoints as our multiplicative decay,
        # slightly different curve (cannot be made identical in SB3)
        exploration_fraction=0.5,
        exploration_initial_eps=TRAIN_CONFIG["epsilon_start"],
        exploration_final_eps=TRAIN_CONFIG["epsilon_end"],
        learning_starts=batch_size,
        verbose=0,
        # unresolvable differences vs our DQN without subclassing SB3:
        #   loss    -- SB3 uses Huber loss, ours uses MSE
        #   Bellman -- SB3 uses Double DQN, ours uses standard DQN
    )

    callback = HexxedProgressCallback(tracker, total_steps)
    model.learn(total_timesteps=total_steps, callback=callback, progress_bar=False)

    return tracker.learning_speed_score(total_steps), tracker
