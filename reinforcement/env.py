# import gym stuff
import sys
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary

# import TM stuff
from tminterface.client import run_client
from tminterface.interface import TMInterface

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
        agent.make_move(TMInterface, action)

        # update state
        self.state = agent.export_data()

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
        agent.restart(TMInterface)
        self.state = np.zeros(8).astype(int)
        self.episode_length = 30
        self.last_sum_of_chekpoints = 0
        self.current_sum_of_checkpoints = 0
        return self.state



agent = TMAgent()
env = TMEnv()

# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(agent, server_name)

if __name__ == '__main__':
    main()



'''Testing the env
episodes = 10
for episode in range(1, episodes + 1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        score += reward
    
    print(f'Episode:{episode} Score:{score}')

env.close()
'''
    


