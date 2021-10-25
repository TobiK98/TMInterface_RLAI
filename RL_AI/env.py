from gym import Env
from gym.spaces import Discrete, Box, Dict, MultiBinary


class TMEnv(Env):
    def __init__(self):
        # actions: left, straight, right
        self.action_space = Discrete(3)

        # observations: speed, position, velocity, checkpoint_state
        self.observation_space = Dict({'position': Box(),
                                        'velocity': Box(),
                                        'checkpoint_state': MultiBinary()})
                                        
        # length of each period
        self.episode_length = 30
        pass

    def step():
        pass

    def render():
        pass

    def reset():
        pass