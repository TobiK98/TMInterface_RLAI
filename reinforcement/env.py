# import gym stuff
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary

# import agent
from agent import TMAgent

# other imports
import numpy as np


class TMEnv(Env):
    # initialize environment
    def __init__(self):
        # actions: left, straight, right
        self.action_space = Discrete(3)

        # observations: reached checkpoints
        self.observation_space = MultiBinary(8)

        self.state = np.zeros(8).astype(int)
        self.episode_length = 30
        self.last_sum_of_chekpoints = 0
        self.current_sum_of_checkpoints = 0

    def step(self, action):
        # make the next move
        TMAgent.make_move(action)

        # update state
        self.state = TMAgent.receive_data()

        # decrease episode_length
        self.episode_length -= 1

        # calculate rewards
        self.last_sum_of_chekpoints = self.current_sum_of_checkpoints
        self.current_sum_of_checkpoints = sum(self.state)
        if self.last_sum_of_chekpoints != self.current_sum_of_checkpoints:
            reward = +100
        else:
            reward = -0.1

        # check whether agent reached the goal
        if self.state[-1] == 1:
            done = True
        else:
            done = False

        info = {}

        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        TMAgent.reset()
        self.state = np.zeros(8).astype(int)
        self.episode_length = 30
        self.last_sum_of_chekpoints = 0
        self.current_sum_of_checkpoints = 0


env = TMEnv()