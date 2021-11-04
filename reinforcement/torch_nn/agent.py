# import TM stuff
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# import model
from model import Linear_QNet, QTrainer

# import others
import sys
import random
import numpy as np
import torch
from collections import deque

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Q_Agent(Client):
    def __init__(self) -> None:
        super(Q_Agent, self).__init__()
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.episodes = 0
        self.record = 0
        self.model = Linear_QNet(3, 256, 4)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        self.covered_track_prev = 0
        self.covered_track = 0
        self.game_over = False
        self.score = 0

    # register to TM
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0:
            # get old state
            state_old = self.get_state(iface)

            # get move
            final_move = self.get_action(state_old)

            # perform move and get new state
            reward = self.play_step(final_move, iface)
            done = self.game_over
            self.score += reward
            state_new = self.get_state(iface)

            # train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            self.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory
                self.reset(iface)
                self.episodes += 1
                self.train_long_memory()

                if self.score > self.record:
                    self.record = self.score
                    self.model.save()

                print(f'Game: {self.episodes}, Score: {self.score}, Record: {self.record}')

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            self.game_over = True
            iface.prevent_simulation_finish()

    def reset(self, iface: TMInterface):
        # reset self
        iface.give_up()
        self.covered_track_prev = 0
        self.covered_track = 0
        self.game_over = False
        self.score = 0

    def play_step(self, action, iface: TMInterface):
        # perform the given action
        if action == 0:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=False)
        if action == 1:
            iface.set_input_state(left=True, accelerate=True, right=False, brake=False)
        if action == 2:
            iface.set_input_state(left=False, accelerate=True, right=True, brake=False)
        if action == 3:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=True)

        # calculate reward for performed action
        reward = 0
        self.covered_track_prev = self.covered_track
        self.covered_track = np.round(iface.get_simulation_state().position[2])
        if self.covered_track > self.covered_track_prev:
            reward += 10
        else:
            reward += -10

        return reward

    def get_state(self, iface: TMInterface):
        return iface.get_simulation_state().position

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploration
        self.epsilon = 90 - self.episodes
        final_move = [0, 0, 0, 0]
        
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move.index(1)


agent = Q_Agent()

# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(agent, server_name)

if __name__ == '__main__':
    main()
