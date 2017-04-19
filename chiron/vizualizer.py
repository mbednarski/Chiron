import abc
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt

from chiron.monitor import PersistentBuffer


class Combiner:
    def __init__(self, root_directory):
        self._root_directory = root_directory

    def _get_collections(self, root):
        ls = [(x, os.path.join(root, x)) for x in os.listdir(root)]
        ls = filter(lambda x: os.path.isdir(x[1]), ls)
        return dict(list(ls))

    def select_latest_session(self):
        ls = [(x, os.path.join(self._root_directory, x)) for x in os.listdir(self._root_directory)]
        ls = filter(lambda x: os.path.isdir(x[1]), ls)
        parsed = [
            (x[0], x[1],
             datetime.datetime.strptime(x[0], '%Y-%m-%d_%H_%M_%S'))
            for x
            in ls
        ]
        parsed.sort(key=lambda x: x[2])

        latest = parsed[-1]

        self._session_location = latest[1]
        self._session_datetime = latest[2]
        self._series_locations = self._get_collections(self._session_location)

    def get_series(self, friendly_name, name):
        loc = self._series_locations[name]
        s = OfflineSeries(friendly_name, loc)
        return s


class SeriesBase(abc.ABC):
    def __init__(self, friendly_name=None):
        self.friendly_name = friendly_name

    @abc.abstractmethod
    def get_data(self):
        pass


class RollingAverageSeries(SeriesBase):
    def __init__(self, friendly_name, parent_series, window=100):
        super().__init__(friendly_name)
        self.raw_data = parent_series.get_data()
        self.window = window

    def get_data(self):
        avgs = np.zeros_like(self.raw_data)
        for i in range(avgs.shape[0]):
            avgs[i] = np.mean(self.raw_data[i - self.window:i])
        return avgs


class OfflineSeries(SeriesBase):
    def __init__(self, friendly_name, location):
        super().__init__(friendly_name)
        self.location = location
        self.data = PersistentBuffer.read_location(location)

    def get_data(self):
        return self.data


class Visualizer:
    def __init__(self, figure, subplots=None):
        self.figure = figure
        self.subplots = subplots if subplots is not None else [111]
        self.series_to_plot = []
        for sp in subplots:
            self.figure.add_subplot(sp)
        self.axes = self.figure.get_axes()

    def append(self, data, axis=1):
        self.series_to_plot.append((data, axis))

    def plot(self):
        for s, axis in self.series_to_plot:
            self.axes[axis].plot(s.get_data(), label=s.friendly_name)
            self.axes[axis].legend()

    def save(self):
        self.figure.savefig('plot.png')
        self.figure.show()


def dd():
    fig = plt.figure()
    v = Visualizer(fig, [211, 212])
    root_dir = r'C:\p\github\Chiron\chiron\agents\monitor'
    c = Combiner(root_dir)
    c.select_latest_session()
    x = c.get_series('epsilon', 'epsilon')
    r = c.get_series('reward', 'episode_reward')
    a = RollingAverageSeries('Average 100', r, window=100)
    v.append(x, 0)
    v.append(r, 1)
    v.append(a, 1)
    v.plot()
    v.save()
    plt.show()


if __name__ == '__main__':
    dd()
