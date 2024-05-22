import os
import pickle
import numpy as np


def read_pickle(filepath, suffix=".pkl"):
    assert os.path.splitext(filepath)[1] == suffix
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def write_pickle(result, filepath):
    with open(filepath, 'wb') as f:
        pickle.dump(result, f)
