import os
from datetime import datetime


def make_now_path(basedir=None):
    path = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
    if basedir is not None:
        path = os.path.join(basedir, path)
    os.makedirs(path, exist_ok=True)
    return path
