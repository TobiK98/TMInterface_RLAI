# import TM stuff
import sys
from time import time
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# import gym stuff
import sys
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary

# import stable-baselines
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import A2C

# import others
import numpy as np


class TMEnv(Client, Env):
    def __init__(self) -> None:
        super(TMEnv, self).__init__()

        # game 
        self.checkpoint_states = None
        self.speed = 0

        # environment, actions: left, straight, right; observations: checkpoint states
        self.action_space = Discrete(4)
        self.observation_space = MultiBinary(8)
        self.state = np.zeros(8).astype(int)
        self.episode_length = 30
        self.checkpoints_reached = np.zeros(8).astype(int)
        self.checkpoints_reached_previous = np.zeros(8).astype(int)
        self.current_reward = 0
        self.current_done = False
        self.current_info = {}
        self.score = 0

    # register to TM
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    # step
    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0:
            # get data from game and set it into variables
            game_data = iface.get_simulation_state()
            self.speed = game_data.display_speed
            checkpoint_data = iface.get_checkpoint_state()
            self.checkpoint_states = checkpoint_data.cp_states

            # get the next action
            action = self.action_space.sample()
            #action = QLearning(self.state)

            # perform a move
            if action == 0:
                iface.set_input_state(left=True, accelerate=True, right=False, brake=False)
            if action == 1:
                iface.set_input_state(left=False, accelerate=True, right=False, brake=False)
            if action == 2:
                iface.set_input_state(left=False, accelerate=True, right=True, brake=False)
            if action == 3:
                iface.set_input_state(left=False, accelerate=True, right=False, brake=True)
            
            # update state
            self.state = self.checkpoint_states

            # decrease length of episode
            self.episode_length -= 1

            # calculate reward
            reward = 0
            self.checkpoints_reached_previous = self.checkpoints_reached
            self.checkpoints_reached = self.state
            if sum(self.checkpoints_reached) > sum(self.checkpoints_reached_previous):
                reward += +100
            else:
                reward += -0.5

            if _time > 5000 and self.speed > 50:
                reward += +1
            else:
                reward += -0.5
            
            self.current_reward = reward

            # check whether game is done
            if self.state[-1] == 1:
                done = True
            else:
                done = False
            self.current_done = done

            # add info if necessary
            info = {}
            self.current_info = info

            self.score += self.current_reward
            print(f'REWARD: {self.score}')

            #self.step()

    def step(self, action):
        self.on_run_step(TMInterface, time, action)
        return self.state, self.current_reward, self.current_done, self.current_info

    # let game play when goal reached
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            iface.prevent_simulation_finish()
            iface.give_up()
            self.state = np.zeros(8).astype(int)
            self.episode_length = 30
            self.checkpoints_reached = np.zeros(8).astype(int)
            self.checkpoints_reached_previous = np.zeros(8).astype(int)
            self.current_reward = 0
            self.current_done = False
            self.current_info = {}
            self.score = 0
            self.reset()

    # reset the level
    def reset(self):
        return self.state

    # generate random action
    def get_action(self):
        return self.action_space.sample()

    def render():
        pass


env = TMEnv()

def model_learn():
    model = A2C('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=10000)
    model.save('A2C_TM')


# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(env, server_name)

# test the agent
def test():
    episodes = 10

    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            obs, reward, done, info = env.step()
            score += reward
        
        print(f'Episode:{episode} Score:{score}')

    env.close()


if __name__ == '__main__':
    main()