import os
import pickle
import numpy as np


def read_pickle(filepath, suffix=".pkl"):
    assert os.path.splitext(filepath)[1] == suffix