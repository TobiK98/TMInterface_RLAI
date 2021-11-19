import numpy as np
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
from agent import Agent
from utils import plot_learning
import sys


class Game(Client):
    def __init__(self) -> None:
        super(Game, self).__init__()
        self.agent = Agent(gamma=0.99, epsilon=1.0, batch_size=600, n_actions=4, eps_end=0.001, input_dims=[6], lr=0.001)
        self.score = 0
        self.scores = []
        self.history = []
        self.epsilon_over_time = []
        self.speed = 0
        self.speed_prev = 0
        self.current_game = 0
        self.cp_sum = 4
        self.cp_reached = 0
        self.cp_reached_prev = 0
        self.avg_scores = []
        self.end_position = np.array([502.36, 9.36, 559.93])
        self.current_distance = 0
        self.prev_distance = 0
        self.max_distance = 270

    def reset(self, iface: TMInterface):
        if self.current_game % 100 == 0:
            plot_learning(np.arange(self.current_game), self.scores, self.avg_scores, self.epsilon_over_time, self.current_game)
        self.score = 0
        self.speed = 0
        self.speed_prev = 0
        self.cp_reached = 0
        self.cp_reached_prev = 0
        self.current_distance = 0
        self.prev_distance = 0
        iface.give_up()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0 and _time % 50 == 0:
            observation = self.get_observation(iface)
            action = self.agent.choose_action(observation)
            self.perform_action(action, iface)

            reward = self.calculate_reward(observation, iface, _time)
            observation_ = self.get_observation(iface)
            done = True if _time > 20000 or sum(iface.get_checkpoint_state().cp_states) == self.cp_sum or (_time > 1000 and iface.get_simulation_state().display_speed <= 20) else False

            self.score += reward

            self.agent.store_transitions(observation, action, reward, observation_, done)
            self.agent.learn()

            if done:
                self.scores.append(self.score)
                self.history.append(self.agent.epsilon)
                avg_score = np.mean(self.scores)
                self.avg_scores.append(avg_score)
                self.epsilon_over_time.append(self.agent.epsilon)
                print(f'Game: {self.current_game}, Score: {self.score}, Avg-Score: {avg_score}, Epsilon: {self.epsilon_over_time[-1]}')
                self.current_game += 1
                self.reset(iface)

    def get_observation(self, iface: TMInterface):
        position = np.round(iface.get_simulation_state().position, 1).astype(np.float32)
        velocity = np.round(iface.get_simulation_state().velocity, 1).astype(np.float32)
        return np.append(position, velocity)

    def perform_action(self, action, iface: TMInterface):
        if action == 0:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=False)
        if action == 1:
            iface.set_input_state(left=True, accelerate=True, right=False, brake=False)
        if action == 2:
            iface.set_input_state(left=False, accelerate=True, right=True, brake=False)
        if action == 3:
            iface.set_input_state(left=False, accelerate=True, right=False, brake=True)

    def calculate_reward(self, observation, iface: TMInterface, _time: int):
        reward = 0
        '''
        # every half second
        if _time % 500 == 0:
            # coming closer to goal
            self.prev_distance = self.current_distance
            self.current_distance = self.calculate_distance(np.split(observation, 2)[0])
            if self.current_distance > self.prev_distance:
                reward += -15000
            else:
                reward += 500

            # going on the wall
            self.frame_speed_prev = self.frame_speed
            self.frame_speed = iface.get_simulation_state().display_speed
            if self.frame_speed <= self.frame_speed_prev + 5:
                reward += -5000
            else:
                reward += 500

        # hitting something
        if self._check_crash(iface):
            reward += -2500
        else:
            reward += iface.get_simulation_state().display_speed * 2

        # standing on wall or only breaking
        if _time > 2000 and self.speed < 40:
            reward += -20000            
        
        # reaching checkpoints
        self.cp_reached_prev = self.cp_reached
        self.cp_reached = sum(iface.get_checkpoint_state().cp_states)
        if self.cp_reached > self.cp_reached_prev:
            if self.cp_reached == self.cp_sum:
                reward += +50000
            else:
                reward += +(2000 * self.cp_reached)
        '''
        calc = (self.calculate_distance(iface.get_simulation_state().position) / self.max_distance)
        current_divider = calc if calc > 0.2 else 0.2
        self.speed = iface.get_simulation_state().display_speed
        if _time > 1000 and self.speed <= 20:
            reward += -250000 * current_divider
        if _time > 100:
            self.prev_distance = self.current_distance
            self.current_distance = self.calculate_distance(np.split(observation, 2)[0])
            if self.current_distance < self.prev_distance:
                reward += +((self.prev_distance - self.current_distance) * 5)
            else:
                reward += -1000
            if self.getting_faster(iface):
                reward += iface.get_simulation_state().display_speed 
            else:
                reward += -250
            if self._check_crash(iface):
                reward += -1000
        self.cp_reached_prev = self.cp_reached
        self.cp_reached = sum(iface.get_checkpoint_state().cp_states)
        if self.cp_reached > self.cp_reached_prev:
            if self.cp_reached == self.cp_sum:
                reward += +5000
            else:
                reward += +(500 * self.cp_reached)
        
        return reward

    def _check_crash(self, iface: TMInterface):
        self.speed_prev = self.speed
        self.speed = iface.get_simulation_state().display_speed
        if self.speed < self.speed_prev - 5:
            return True
        return False

    def getting_faster(self, iface: TMInterface):
        self.speed_prev = self.speed
        self.speed = iface.get_simulation_state().display_speed
        if self.speed >= self.speed_prev:
            return True
        else:
            return False

    def calculate_distance(self, location):
        distance = np.sqrt(np.power((self.end_position[0] - location[0]), 2) + 
                            np.power((self.end_position[1] - location[1]), 2) +
                            np.power((self.end_position[2] - location[2]), 2))
        return distance

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            iface.prevent_simulation_finish()


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(Game(), server_name)

if __name__ == '__main__':
    main()
