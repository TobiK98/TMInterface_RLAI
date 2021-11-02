# import TM stuff
import sys
from time import time
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client


# import others
import numpy as np


class Q_Agent(Client):
    def __init__(self) -> None:
        super(Q_Agent, self).__init__()
        self.state = np.zeros(8)
        self.actions = np.arange(4)
        self.q = np.zeros([9, 4])
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.epsilon_decay = 0.9995
        self.discount = 0.95
        self.current_episode = 0
        self.episodes = 100
        self.episode_reward = 0
        self.rewards = np.array([])
        self.prev_state = np.zeros(8)
        self.best_reward = 0
        self.best_episode = 0

    # register to TM
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    # step
    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0:
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.actions)
            else:
                action = np.argmax(self.q[sum(self.state).astype(int)])

            if action == 0:
                iface.set_input_state(left=True, accelerate=True, right=False, brake=False)
            if action == 1:
                iface.set_input_state(left=False, accelerate=True, right=False, brake=False)
            if action == 2:
                iface.set_input_state(left=False, accelerate=True, right=True, brake=False)
            if action == 3:
                iface.set_input_state(left=False, accelerate=True, right=False, brake=True)

            self.prev_state = self.state
            cp_data = iface.get_checkpoint_state()
            self.state = np.array(cp_data.cp_states)

            reward = 0
            if sum(self.state) > sum(self.prev_state):
                reward += +25
            else:
                reward = -0.5
            if sum(self.state) == 8:
                reward += +250
            self.episode_reward += reward

            current_q = self.q[sum(self.state).astype(int)][action]
            max_future_q = np.max(self.q[sum(self.state).astype(int)])
            new_q = (1 - self.learning_rate) * current_q + self.learning_rate * (reward + self.discount * max_future_q)
            self.q[sum(self.state).astype(int)][action] = new_q

            if _time == 10000 or sum(self.state) == 8:
                if self.episode_reward > self.best_reward:
                    self.best_reward = self.episode_reward
                    self.best_episode = self.current_episode
                print(f'===== Finished episode {self.current_episode} =====')
                print(f'Episode reward: {self.episode_reward}')
                self.rewards = np.append(self.rewards, self.episode_reward)
                print(f'Average reward: {np.mean(self.rewards)}')
                print(f'Best episode so far: {self.best_episode} with a reward of {self.best_reward}\n')
                self.state = np.zeros(8)
                self.current_episode += 1
                self.episode_reward = 0
                self.epsilon *= self.epsilon_decay
                iface.give_up()

            

    # let game play when goal reached
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()



agent = Q_Agent()

# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(agent, server_name)

if __name__ == '__main__':
    main()