import numpy as np
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from agent import Agent
from utils import plot_learning
import sys


class Game(Client):
    def __init__(self) -> None:
        super(Game, self).__init__()
        self.agent = Agent(gamma=0.99, epsilon=1.0, batch_size=1028, n_actions=4, eps_end=0.001, input_dims=[3], lr=0.001)
        self.score = 0
        self.scores = []
        self.history = []
        self.n_games = 500
        self.speed = 0
        self.speed_prev = 0
        self.current_game = 0
        self.cp_sum = 6
        self.plotting_done = False
        self.cp_reached = 0
        self.cp_reached_prev = 0
        self.avg_scores = []

    def reset(self, iface: TMInterface):
        if self.current_game % 100 == 0:
            plot_learning(np.arange(self.current_game), self.scores, self.avg_scores, self.current_game)
        self.score = 0
        self.speed = 0
        self.speed_prev = 0
        self.cp_reached = 0
        self.cp_reached_prev = 0
        iface.give_up()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0:
            observation = np.round(iface.get_simulation_state().position, 2).astype(np.float32)
            action = self.agent.choose_action(observation)
            self.perform_action(action, iface)

            observation_ = np.round(iface.get_simulation_state().position, 2).astype(np.float32)
            reward = self.calculate_reward(iface, _time)
            done = True if _time > 30000 or (_time > 1000 and iface.get_simulation_state().display_speed < 10) or sum(iface.get_checkpoint_state().cp_states) == self.cp_sum else False

            self.score += reward

            self.agent.store_transitions(observation, action, reward, observation_, done)
            self.agent.learn()

            if done:
                self.scores.append(self.score)
                self.history.append(self.agent.epsilon)
                avg_score = np.mean(self.scores)
                self.avg_scores.append(avg_score)
                print(f'Game: {self.current_game}, Score: {self.score}, Avg-Score: {avg_score}, Epsilon: {self.agent.epsilon}')
                self.current_game += 1
                self.reset(iface)

    def perform_action(self, action, iface: TMInterface):
        if action == 0:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=False)
        if action == 1:
            iface.set_input_state(left=True, accelerate=True, right=False, brake=False)
        if action == 2:
            iface.set_input_state(left=False, accelerate=True, right=True, brake=False)
        if action == 3:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=True)

    def calculate_reward(self, iface: TMInterface, _time: int):
        reward = 0
        if self.speed < self.speed_prev - 10:
            reward += -20000
        else:
            reward += (+_time / 1000) + (iface.get_simulation_state().display_speed / 5)
        self.cp_reached_prev = self.cp_reached
        self.cp_reached = sum(iface.get_checkpoint_state().cp_states)
        if self.cp_reached > self.cp_reached_prev:
            if self.cp_reached == self.cp_sum:
                reward += 40000
            else:
                reward += 1000 * self.cp_reached
        return reward

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            self.game_over = True
            iface.prevent_simulation_finish()


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Game(), server_name)

if __name__ == '__main__':
    main()
