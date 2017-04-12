import zmq
import time
import timeit
import numpy as np

port = 6587
context = zmq.Context()
socket = context.socket(zmq.PAIR)
print('Connecting...')
socket.connect("tcp://localhost:{}".format(port))
# print('Connected')

i = 0

# msg = socket.recv_string()
# print('Confirmation :{}'.format(msg))

while True:
    print('Sending {}...'.format(i))
    socket.send_pyobj([i, i + np.random.uniform(low=-1,high=1)], zmq.NOBLOCK)
    print('Sent')
    i += 1
    time.sleep(0.1)