from threading import Thread
import time

def func1():
    while True:
        print(f'Thread 1: {time.localtime().tm_sec}\n')

def func2():
    while True:
        print(f'Thread 2: {time.localtime().tm_sec}\n')


a = Thread(target = func1)
b = Thread(target = func2)
a.start()
b.start()

