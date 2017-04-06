import os
from time import time
import json

import numpy as np

import misc


class PersistentBuffer(object):
    def __init__(self, name, basedir, shape=(1,), save_interval=1.0):
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

    def have_to_dump(self):
        if self.pointer >= self.MAX_BUFFER:
            return True

        if time() - self.last_save_time > self.save_interval:
            return True

        return False

    def append(self, value):
        if self.have_to_dump():
            self.dump()

        self.buffer[self.pointer] = value
        self.pointer += 1


class Monitor(object):
    def __init__(self, basedir='monitor', save_interval=10.0):
        self.save_interval = save_interval
        self.buffers = {}
        self.rootdir = misc.make_now_path(basedir)

    def add_buffer(self, name):
        self.buffers[name] = PersistentBuffer(name, basedir=self.rootdir)

    def append_episode(self, name, value):
        self.buffers[name].append(value)

    def append_episode_dict(self, data):
        for k, v in data.items():
            self.buffers[k].append(v)


    def dump(self):
        for _, v in self.buffers.items():
            v.dump()

    def write_info(self, agent, env):
        data = {'name': agent.name}
        data.update(agent.get_parameters())

        with open(os.path.join(self.rootdir, 'agent.json'), 'w') as f:
            json.dump(data, f, indent=4)



