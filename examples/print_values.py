from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys
import numpy as np

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')

    def on_run_step(self, iface: TMInterface, _time: int):
        if _time >= 0:
            state = iface.get_simulation_state()
            check = iface.get_checkpoint_state()

            print(
                f'Time: {_time}\n' 
                f'X: {np.round(state.position[2])}\n'
                f'Y: {np.round(state.position[0])}\n'
            )


def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()
