"""
hexxed.py — Hexxed puzzle game environment (professor's implementation, unmodified).

Observation: 72-dim float vector (6×12 hex grid, flattened).
Actions:      Discrete(7) — 6 rotations + 1 collect action.
Reward:       squared match-length when collecting (1–36 per pattern).
"""

import os
import random
from gym import core, spaces
import numpy as np


class hexxed(core.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.curr_wave   = 1
        self.curr_act    = 0
        self.num_attempts = 0
        self.wave_reward = 0
        self.max_reward  = 0
        self.subwave_num = 0

    def ready(self, num_vertices=6, step_per_pattern=6, levels=6,
              shuffle_patterns=True, random_rolls=True,
              normalize_reward=False, perfect_bonus=False, render_mode=1):
        self.grid_x       = num_vertices
        self.cut_off      = step_per_pattern
        self.num_levels   = levels
        self.grid_y       = 2 * num_vertices
        self.shuffle_patterns = shuffle_patterns
        self.random_rolls     = random_rolls
        self.normalize_reward = normalize_reward
        self.render_mode      = render_mode
        self.perfect_bonus    = 36 if perfect_bonus else 0
        self.pattern_list     = self.read_patterns(range(self.curr_wave))
        self.action_space      = spaces.Discrete(num_vertices + 1)
        grid_size              = self.grid_x * self.grid_y
        self.observation_space = spaces.Box(
            np.zeros(grid_size, dtype=np.float32),
            np.ones(grid_size,  dtype=np.float32),
            dtype=np.float32,
        )
        self.subwave_id  = []
        self.reward_mean = []
        self.level_num   = []
        self.attempt_num = []
        self.reward_hist = []
        self.level_hist  = []
        self.action_hist = []

    def read_patterns(self, num_targets):
        _here = os.path.dirname(os.path.abspath(__file__))
        all_patterns = np.loadtxt(os.path.join(_here, 'spawn.txt'), 'int', delimiter=',')
        patterns = []
        for i in num_targets:
            level_patterns = all_patterns[all_patterns[..., 1] == i + 1, ...]
            level_patterns = np.append(
                level_patterns[..., 0].reshape((-1, i + 1)),
                level_patterns[::i + 1, 3].reshape((-1, 1)), 1,
            )
            if self.shuffle_patterns:
                np.random.shuffle(level_patterns)
            patterns.extend(level_patterns.tolist())
        return patterns

    def step(self, action):
        if self.render_mode:
            self.render()
        self.curr_act = action
        self.action_hist.append(action)
        self.subwave_id.append(self.subwave_num)
        reward = 0
        done   = 0
        if action < 6:
            self.step_grid(action)
        else:
            if self.ismoving and np.sum(self.grid[0, self.cut_off:]):
                reward = (np.where(self.grid[0, self.cut_off:])[0][0] + 1) ** 2
                self.grid[0, :] = 0
            else:
                self.step_grid(0)
        self.ismoving = True
        if self.normalize_reward:
            reward /= 36.0
        done = np.sum(self.grid) == 0
        self.subwave_reward += max(0, reward)
        if done:
            if self.subwave_reward == self.subwave_max:
                self.max_reward += self.perfect_bonus * self.curr_wave
                reward += self.perfect_bonus * self.curr_wave
            self.max_reward  += self.pattern_list[self.subwave_num][-1]
            self.subwave_num  = (self.subwave_num + 1)
        self.wave_reward += max(0, reward)
        self.reward_hist.append(self.subwave_reward)
        self.reward_mean.append(self.wave_reward)
        self.level_hist.append(self.curr_wave)
        self.attempt_num.append(self.num_attempts)
        if self.render_mode:
            print(self.curr_act, end=' ')
            print(self.curr_wave, end=' ')
            print(self.num_attempts, end=' ')
            print(self.subwave_reward)
        return self.grid.flatten(), reward, done, {}

    def step_grid(self, vert):
        if self.ismoving:
            dist = max(1, min(vert, 6 - vert))
        else:
            dist = 1
        self.grid = np.roll(np.roll(self.grid, -vert, 0), dist, 1)
        self.grid[:, :dist] = 0

    def reset(self):
        if (self.curr_wave < 6 and self.wave_reward < self.max_reward - len(self.pattern_list) * 15) or \
           (self.curr_wave == 6 and self.wave_reward < self.max_reward - len(self.pattern_list) * 6):
            self.reset_helper()
        elif self.subwave_num == len(self.pattern_list):
            self.reset_helper()
            if self.curr_wave == self.num_levels:
                self.curr_wave = 1
            else:
                self.curr_wave = min(self.curr_wave + 1, self.num_levels)
        if self.subwave_num == 0:
            self.pattern_list = self.read_patterns(range(self.curr_wave - 1, self.curr_wave))
            self.wave_reward  = 0
            self.max_reward   = 0
            self.num_attempts += 1
        self.ismoving      = False
        self.subwave_max   = self.pattern_list[self.subwave_num][-1]
        self.subwave_reward = 0
        self.grid = np.zeros(shape=(self.grid_x, self.grid_y))
        for i in range(len(self.pattern_list[self.subwave_num]) - 1):
            self.grid[self.pattern_list[self.subwave_num][i]][self.cut_off - i - 1] = 1
        if self.random_rolls:
            self.grid = np.roll(self.grid, random.randint(0, self.grid_x), 0)
            if random.randint(0, 2):
                self.grid = np.roll(np.flip(self.grid, 0), 1, 0)
        return self.grid.flatten()

    def reset_helper(self):
        self.subwave_num = 0

    def render(self, mode='human', close=False):
        if close:
            return
        hexagon = np.zeros(self.grid_x)
        print(' ', end='')
        for x in range(self.grid_x):
            y = np.where(self.grid[x, :])[0]
            if np.size(y):
                hexagon[x] = y[0]
        for x in range(self.grid_x):
            out_val = int(hexagon[x] - 5)
            if out_val == -5:
                print('.', end='  ')
            elif out_val < 0:
                print(out_val, end=' ')
            else:
                print(out_val, end='  ')
