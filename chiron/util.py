import os
from datetime import datetime
import shutil


def pairwise(seq):
    seq = iter(seq)
    first = next(seq)
    second = next(seq)
    yield (first, second)

    try:
        while True:
            first = second
            second = next(seq)
            yield (first, second)
    except StopIteration:
        # that's OK
        return



def make_now_path(basedir=None):
    path = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    if basedir is not None:
        path = os.path.join(basedir, path)
    os.makedirs(path, exist_ok=True)
    return path


def ensure_empty_dir(dirname):
    try:
        shutil.rmtree(dirname)
    except FileNotFoundError:
        # dir does not exist
        pass

    os.mkdir(dirname)
