# import TM stuff
from tminterface.interface import TMInterface
from tminterface.client import Client, run_client

# import environment
# from env import TMEnv

# import others
import numpy as np
import sys



class TMAgent(Client):
    # initialize game
    def __init__(self) -> None:
        super(TMAgent, self).__init__()
        self.checkpoint_states = None
        self.position = None
        self.velocity = None

    # register to TM
    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    # read TM data
    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            state = iface.get_simulation_state()
            check = iface.get_checkpoint_state()
            self.export_data(check.cp_states)

    # transform data and export it to agent
    def export_data(self, data):
        self.checkpoint_states = data
        return np.array(self.checkpoint_states).astype(int)

    # let game play when goal reached
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached checkpoint {current}/{target}')
        if current == target:
            iface.prevent_simulation_finish()

    # restart level
    def restart(self, iface: TMInterface):
        iface.give_up()

    # make the next move
    def make_move(self, iface: TMInterface, action):
        if action == 0:
            move = iface.set_input_state(left=True)
        if action == 1:
            move = iface.set_input_state(accelerate=True)
        if action == 2:
            move = iface.set_input_state(right=True)
        return move



# connect to TM
def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(TMAgent(), server_name)


if __name__ == '__main__':
    main()