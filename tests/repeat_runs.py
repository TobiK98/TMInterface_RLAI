from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import pandas as pd

class MainClient(Client):
    def __init__(self) -> None:
        self.race_time = 0
        self.game_num = 0
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        self.race_time = _time 

        if _time == 0:
            print(f'Starting game number {self.game_num}')
    
    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        print(f'Reached Checkpoint {current}/{target}')

        if current == target:
            print(f'Finished race at {self.race_time}')
            iface.prevent_simulation_finish()
            iface.give_up()
            self.game_num += 1


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()