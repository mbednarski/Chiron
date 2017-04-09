import os
from datetime import datetime
import shutil


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
