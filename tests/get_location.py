from tminterface.interface import TMInterface
from tminterface.client import Client, run_client
import sys

class MainClient(Client):
    def __init__(self) -> None:
        super(MainClient, self).__init__()

    def on_registered(self, iface: TMInterface) -> None:
        print(f'Registered to {iface.server_name}')
        print(iface.get_simulation_state().position)

    def on_checkpoint_count_changed(self, iface: TMInterface, current: int, target: int):
        if current == target:
            print(iface.get_simulation_state().position)
            iface.prevent_simulation_finish()

def main():
    server_name = f'TMInterface{sys.argv[1]}' if len(sys.argv) > 1 else 'TMInterface0'
    print(f'Connecting to {server_name}...')
    run_client(MainClient(), server_name)


if __name__ == '__main__':
    main()
