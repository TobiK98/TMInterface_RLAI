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

MAX_MEMORY = 1_000_000
BATCH_SIZE = 10_000
LEARNING_RATE = 0.001

class Q_Agent(Client):
    def __init__(self) -> None:
        super(Q_Agent, self).__init__()
        self.start_point = [757, 10, 624]
        self.end_point = [182, 9, 624]
        self.total_distance = self.calculate_distance(self.start_point)
        self.covered_track_prev = 0
        self.covered_track = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.episodes = 0
        self.record = 0
        self.model = Linear_QNet(6, 256, 4)
        self.trainer = QTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)
        self.game_over = False
        self.score = 0
        self.max_episode_time = 30000
        self.checkpoints_reached = 0
        self.checkpoints_reached_prev = 0
        self.number_of_checkpoints = 6
        self.speed = 0
        self.speed_prev = 0
        self.time_without_crash = 0

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
            reward = self.play_step(final_move, iface, _time)
            done = self.game_over
            self.score += reward
            state_new = self.get_state(iface)

            # train short memory
            self.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            self.remember(state_old, final_move, reward, state_new, done)

            if _time > self.max_episode_time:
                self.game_over = True

            if done:
                # train long memory
                self.episodes += 1
                self.train_long_memory()

                if self.score > self.record:
                    self.record = self.score
                    self.model.save()

                print(f'Game: {self.episodes}, Score: {self.score}, Record: {self.record}')
                self.reset(iface)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            self.game_over = True
            iface.prevent_simulation_finish()

    def reset(self, iface: TMInterface):
        # reset self
        self.covered_track_prev = 0
        self.covered_track = 0
        self.checkpoints_reached = 0
        self.checkpoints_reached_prev = 0
        self.game_over = False
        self.score = 0
        self.speed = 0
        self.speed_prev = 0
        self.time_without_crash = 0
        iface.give_up()

    def play_step(self, action, iface: TMInterface, _time: int):
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
        self.speed_prev = self.speed
        self.speed = iface.get_simulation_state().display_speed
        if not self.crash():
            reward += _time / 100
            reward += iface.get_simulation_state().display_speed
        else: 
            reward = -10000
            self.game_over = True
        
        '''
        self.covered_track_prev = self.covered_track
        self.covered_track = self.calculate_distance(np.round(iface.get_simulation_state().position).astype(int))
        if self.covered_track > self.covered_track_prev:
            reward += 10
        else:
            reward += -50
        self.checkpoints_reached_prev = self.checkpoints_reached_prev
        self.checkpoints_reached = sum(iface.get_checkpoint_state().cp_states)
        
        if self.checkpoints_reached > self.checkpoints_reached_prev:
            reward += ((self.max_episode_time - _time) / (self.number_of_checkpoints + 1 - self.checkpoints_reached)) / 20 * self.checkpoints_reached
        if self.checkpoints_reached == self.number_of_checkpoints:
            reward += +5000
        if self.hit_wall():
            reward += -10000
        '''

        return reward

    def crash(self):
        did_crash = False
        if self.speed <= (self.speed_prev - 10):
            did_crash = True 
        return did_crash

    def calculate_distance(self, current_location):
        distance = np.sqrt(np.power((self.end_point[0] - current_location[0]), 2) + 
                            np.power((self.end_point[1] - current_location[1]), 2) +
                            np.power((self.end_point[2] - current_location[2]), 2))
        print(distance)
        return distance

    def get_state(self, iface: TMInterface):
        position = np.array(iface.get_simulation_state().position)
        velocity = np.array(iface.get_simulation_state().velocity )
        return np.append(position, velocity).tolist()

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
        
        if random.randint(0, 100) < self.epsilon:
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
            print(move)

        return final_move.index(1)


agent = Q_Agent()

# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(agent, server_name)

if __name__ == '__main__':
    main()
