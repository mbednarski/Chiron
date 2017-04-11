import zmq
import time

import matplotlib.pyplot as plt

port = 6587
context = zmq.Context()
socket = context.socket(zmq.PAIR)
socket.bind('tcp://*:{}'.format(port))

plt.ion()
fig, ax = plt.subplots()

idx = []
data = []

# socket.send_string('Connected')
plt.show()


while True:
    msg = socket.recv_pyobj()
    idx.append(msg[0])
    data.append(msg[1])
    print('Received: {}'.format(msg))
    ax.cla()
    ax.plot(idx, data)
    fig.canvas.draw()
    plt.pause(0.001)

