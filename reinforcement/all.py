# import TM stuff
import sys
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# import gym stuff
import sys
from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary

# import others
import numpy as np



class TMAgent(Client, Env):
    def __init__(self) -> None:
        super(TMAgent, self).__init__()

        # game 
        self.checkpoint_states = None
        self.position = None
        self.velocity = None

        # environment, actions: left, straight, right; observations: checkpoint states
        self.action_space = Discrete(3)
        self.observation_space = Discrete(9)
        self.state = 0
        self.episode_length = 30
        self.checkpoints_reached = 0
        self.checkpoints_reached_previous = 0
        self.current_reward = 0
        self.current_done = False
        self.current_info = {}

    # register to TM
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    # step
    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            # get data from game and set it into variables
            game_data = iface.get_simulation_state()
            checkpoint_data = iface.get_checkpoint_state()
            self.checkpoint_states = checkpoint_data.cp_states

            # get the next action
            action = self.get_action()

            # perform a move
            if action == 0:
                iface.set_input_state(left=True)
            if action == 1:
                iface.set_input_state(accelerate=True)
            if action == 2:
                iface.set_input_state(right=True)
            
            # update state
            self.state = self.checkpoint_states

            # decrease length of episode
            self.episode_length -= 1

            # calculate reward
            self.checkpoints_reached_previous = self.checkpoints_reached
            self.checkpoints_reached = self.state
            if self.checkpoints_reached > self.checkpoints_reached_previous:
                reward = +100
            else:
                reward = -0.1
            self.current_reward = reward

            # check whether game is done
            if self.state == 8:
                done = True
            else:
                done = False
            self.current_done = done

            # add info if necessary
            info = {}
            self.current_info = info

            self.step()

    def step(self):
        return self.state, self.current_reward, self.current_done, self.current_info

    # let game play when goal reached
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            iface.prevent_simulation_finish()

    # reset the level
    def reset(self, iface: TMInterface):
        iface.give_up()
        self.checkpoint_states = None
        self.position = None
        self.velocity = None
        self.state = 0
        self.episode_length = 30
        self.checkpoints_reached = 0
        self.checkpoints_reached_previous = 0
        return self.state

    # generate random action
    def get_action(self):
        return self.action_space.sample()

    def render():
        pass


agent = TMAgent()

# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(agent, server_name)

# test the agent
def test():
    episodes = 10

    for episode in range(1, episodes + 1):
        obs = agent.reset()
        done = False
        score = 0

        while not done:
            obs, reward, done, info = agent.step()
            score += reward
        
        print(f'Episode:{episode} Score:{score}')

    agent.close()


if __name__ == '__main__':
    main()