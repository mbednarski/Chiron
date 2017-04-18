import os
from time import time
import json

import numpy as np
import zmq

from chiron import util


class OnlineBuffer:
    # def __init__(self, name):


    def append(self, value):
        self.socket.send_pyobj([self.name, value], zmq.NOBLOCK)


class PersistentBuffer(object):
    def __init__(self, name, basedir, shape=(1,), save_interval=10.0):
        self.MAX_BUFFER = 10000
        self.save_interval = save_interval
        self.pointer = 0
        self.last_save_time = time()
        self.effective_dir = os.path.join(basedir, name)
        os.makedirs(self.effective_dir, exist_ok=True)
        self.n_file = 0
        self.buffer = np.zeros(shape=(self.MAX_BUFFER,) + shape)

    def dump(self):
        """
        Dumps buffer content into the file and increment file counter with clearing buffer
        """
        fname = os.path.join(self.effective_dir, '{:07d}'.format(self.n_file))
        data = self.buffer[:self.pointer]
        np.save(fname, data)
        self.pointer = 0
        self.n_file += 1
        self.last_save_time = time()

    def _have_to_dump(self):
        if self.pointer >= self.MAX_BUFFER:
            return True

        if time() - self.last_save_time > self.save_interval:
            return True

        return False

    def append(self, value):
        if self._have_to_dump():
            self.dump()

        self.buffer[self.pointer] = value
        self.pointer += 1

    @staticmethod
    def read_location(location):
        files = [os.path.join(location, x) for x in os.listdir(location)]
        files.sort()
        values = None
        for f in files:
            fcontent = np.load(f)
            if values is None:
                values = fcontent
                continue
            values = np.vstack((values, fcontent))

        return values


class Monitor(object):
    def __init__(self, basedir='monitor', save_interval=30.0):
        self.save_interval = save_interval
        self.buffers = {}
        self.online_buffers = {}
        self.rootdir = util.make_now_path(basedir)

        self.port = 6587
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.setsockopt(zmq.LINGER, 20000)
        print('Connecting...')
        self.socket.connect("tcp://localhost:{}".format(self.port))

    def add_buffer(self, name):
        self.buffers[name] = PersistentBuffer(name, basedir=self.rootdir)

    def append_episode(self, name, value):
        self.buffers[name].append(value)
        # self.socket.send_pyobj([name, value], zmq.NOBLOCK)

    def append_episode_dict(self, data):
        for k, v in data.items():
            self.buffers[k].append(v)
            # self.socket.send_pyobj([k, v], zmq.NOBLOCK)

    def close(self):
        self.socket.close()
        self.context.term()

    def dump(self):
        for _, v in self.buffers.items():
            v.dump()

    def write_info(self, agent, env):
        data = {
            'name': agent.name,
            'env': env.spec.id
        }
        data.update(agent.get_parameters())

        with open(os.path.join(self.rootdir, 'agent.json'), 'w') as f:
            json.dump(data, f, indent=4)
