import os
import datetime
import abc

import numpy as np
import matplotlib.pyplot as plt

import sys

import logging

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)


class SeriesTransformer(abc.ABC):
    @abc.abstractmethod
    def transform(self, data):
        pass


class SeriesBase(abc.ABC):
    def __init__(self, name, friendly_name=None):
        self.name = name
        self._location = None
        self.friendly_name = friendly_name if friendly_name is not None else name

    def set_location(self, location):
        self._location = location

    def _read_values(self):
        files = [os.path.join(self._location, x) for x in os.listdir(self._location)]
        files.sort()
        values = None
        for f in files:
            fcontent = np.load(f)
            if values is None:
                values = fcontent
                continue
            values = np.vstack((values, fcontent))

        return values

    @abc.abstractmethod
    def _get_data(self):
        pass

    def plot(self, axis, **kwargs):
        data = self._get_data()
        axis.plot(data, label=self.friendly_name, **kwargs)
        axis.legend()


class RawSeries(SeriesBase):
    def _get_data(self):
        return self._read_values()

    def __init__(self, name, friendly_name=None):
        super().__init__(name, friendly_name)


class RollingAverageSeries(SeriesBase):
    def __init__(self, name, friendly_name=None, window=50):
        super().__init__(name, friendly_name)
        self.window = window

    def _get_data(self):
        data = self._read_values()
        data = self._compute_rolling_average(data, self.window)
        return data

    def _compute_rolling_average(self, array, window):
        avgs = np.zeros_like(array)
        for i in range(avgs.shape[0]):
            avgs[i] = np.mean(array[i - window:i])
        return avgs


class MaxSeries(SeriesBase):
    def _get_data(self):
        data = self._read_values()
        data = self._compute_max(data)
        return data

    def __init__(self, name, friendly_name):
        super().__init__(name, friendly_name)

    def _compute_max(self, data):
        maxs = np.zeros_like(data)
        for i in range(maxs.shape[0]):
            maxs[i] = np.max(data[:i + 1])
        return maxs


class Plotter(object):
    def __init__(self, root_directory, shape=(1, 1)):
        logger.info("")
        self._root_directory = root_directory
        self._session_location = None
        self._session_datetime = None
        self._series_to_plot = []
        self._series_locations = {}
        self.fig, self.axes = plt.subplots(shape[0], shape[1])

    def select_session(self):
        raise NotImplementedError()

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

    def _get_collections(self, root):
        ls = [(x, os.path.join(root, x)) for x in os.listdir(root)]
        ls = filter(lambda x: os.path.isdir(x[1]), ls)
        return dict(list(ls))

    def append(self, series, axis=1):
        series.set_location(self._series_locations[series.name])
        self._series_to_plot.append((series, axis))

    def plot(self):
        for s, axis in self._series_to_plot:
            s.plot(self.axes[axis])
        plt.show()


def main():
    root_dir = r'C:\p\github\Chiron\chiron\agents\monitor'
    p = Plotter(root_dir, shape=(2, 1))
    p.select_latest_session()
    p.append(RollingAverageSeries('episode_reward', '10 avg reward', 10), 0)
    p.append(RollingAverageSeries('episode_reward', '50 avg reward', 50), 0)
    p.append(RollingAverageSeries('episode_reward', '100 avg reward', 100), 0)
    p.append(RollingAverageSeries('episode_reward', '200 avg reward', 200), 0)
    p.append(MaxSeries('episode_reward', 'max reward'), 0)
    p.append(RawSeries('epsilon', 'epsilon'), 1)
    p.plot()


if __name__ == '__main__':
    main()
