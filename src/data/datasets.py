import cv2
import glob
import logging
import os
import pathlib
import pandas as pd
import numpy as np
import json
from sklearn.datasets.base import Bunch
from scipy.io import loadmat
from functools import partial
from joblib import Memory
import sys

from .utils import fetch_and_unpack
from ..paths import data_path, raw_data_path, interim_data_path, processed_data_path

_MODULE = sys.modules[__name__]
_MODULE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)

jlmem = Memory(cachedir=str(interim_data_path))

@jlmem.cache
def load_coil_20():

    fetch_and_unpack('coil-20')

    c20 = Bunch()
    c20['metadata'] = {}
    c20.metadata['filenames'] = []

    feature_vectors = []
    glob_path = interim_data_path / 'coil-20' / 'processed_images' / '*.pgm'
    filelist = glob.glob(str(glob_path))
    for i, filename in enumerate(filelist):
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())
        c20.metadata['filenames'].append(os.path.basename(filename))

    c20['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    c20['data'] = np.vstack(feature_vectors)
    with open(_MODULE_DIR / 'coil-20.txt') as fd:
        c20['DESCR'] = fd.read()
    return c20

@jlmem.cache
def load_coil_100():

    fetch_and_unpack('coil-100')

    c100 = Bunch()
    feature_vectors = []
    glob_path = interim_data_path / 'coil-100' / 'coil-100/' / '*.ppm'
    filelist = glob.glob(str(glob_path))
    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    c100['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    c100['data'] = np.vstack(feature_vectors)
    with open(_MODULE_DIR / 'coil-100.txt') as fd:
        c100['DESCR'] = fd.read()
    return c100

@jlmem.cache
def load_fmnist(kind='train', return_X_y=False):
    '''
    Load the fashion-MNIST dataset
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''

    fetch_and_unpack('f-mnist')

    fmnist = Bunch()

    if kind == 'test':
        kind = 't10k'

    label_path = interim_data_path / 'f-mnist' / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        fmnist['target'] = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    data_path = interim_data_path / 'f-mnist' / f"{kind}-images-idx3-ubyte"
    with open(data_path, 'rb') as fd:
        fmnist['data'] = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(fmnist['target']), 784)
    with open(_MODULE_DIR / 'f-mnist.txt') as fd:
        fmnist['DESCR'] = fd.read()

    if return_X_y:
        return fmnist.data, fmnist.target
    else:
        return fmnist

@jlmem.cache
def load_mnist(kind='train', variant='mnist', return_X_y=False):
    '''
    Load the MNIST dataset
    variant: {'mnist', 'f-mnist'}
        Which variant to load
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''

    fetch_and_unpack('mnist')

    dset = Bunch()

    if kind == 'test':
        kind = 't10k'

    label_path = interim_data_path / variant / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        dset['target'] = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    data_path = interim_data_path / variant / f"{kind}-images-idx3-ubyte"
    with open(data_path, 'rb') as fd:
        dset['data'] = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(dset['target']), 784)
    with open(_MODULE_DIR / f'{variant}.txt') as fd:
        dset['DESCR'] = fd.read()

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset

@jlmem.cache
def load_dataset(dataset_name, return_X_y=False, **kwargs):
    '''Loads a scikit-learn style dataset

    dataset_name:
        Name of dataset to load
    return_X_y: boolean, default=False
        if True, returns (data, target) instead of a Bunch object
    '''

    if dataset_name not in dataset_raw_files:
        raise Exception(f'Unknown Dataset: {dataset_name}')

    dset = dataset_raw_files[dataset_name]['load_function'](**kwargs)

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset

@jlmem.cache
def load_frey_faces(return_X_y=False, filename='frey_rawface.mat'):
    '''
    Load the Frey Faces dataset

    Note, there are no labels associated with this dataset; i.e.
    `target` is a vector of all zeros
    '''

    frey_file = pathlib.Path(fetch_and_unpack('frey-faces')) / filename

    dset = Bunch()
    ff = loadmat(frey_file, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T

    with open(_MODULE_DIR / 'frey-faces.txt') as fd:
        dset['DESCR'] = fd.read()

    dset.data = ff
    dset.target = np.zeros(ff.shape[0])

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset

def write_dataset(path=None, filename="datasets.json", indent=4, sort_keys=True):
    """Write a serialized (JSON) dataset file"""
    if path is None:
        path = _MODULE_DIR
    else:
        path = pathlib.Path(path)

    ds = dataset_raw_files.copy()
    # copy, adjusting non-serializable items
    for key, entry in ds.items():
        func = entry['load_function']
        del(entry['load_function'])
        entry['load_function_name'] = func.func.__name__
        entry['load_function_options'] = func.keywords
    print(ds)
    with open(path / filename, 'w') as fw:
        json.dump(ds, fw, indent=indent, sort_keys=sort_keys)

def read_datasets(path=None, filename="datasets.json"):
    """Read the serialized (JSON) dataset list
    """
    if path is None:
        path = _MODULE_DIR
    else:
        path = pathlib.Path(path)

    with open(path / filename, 'r') as fr:
        ds = json.load(fr)

    # make the functions callable
    for dset_name, dset_opts in ds.items():
        opts = dset_opts.get('load_function_options', {})
        fail_func = partial(unknown_function, dset_opts['load_function_name'])
        func_name = getattr(_MODULE, dset_opts['load_function_name'], fail_func)
        func = partial(func_name, **opts)
        dset_opts['load_function'] = func

    return ds


def unknown_function(args, **kwargs):
    """Placeholder for unknown function_name"""
    raise Exception("Unknown function: {args}, {kwargs}")

dataset_raw_files = read_datasets()

available_datasets = tuple(dataset_raw_files.keys())
