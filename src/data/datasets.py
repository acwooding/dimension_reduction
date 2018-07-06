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
import sys

from ..paths import data_path, raw_data_path, interim_data_path, processed_data_path
from .utils import fetch_file, unpack, fetch_files



_MODULE = sys.modules[__name__]


_MODULE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)


def fetch_and_process(dataset_name, do_unpack=True):
    '''Fetch and process datasets to their usable form

    dataset_name:
        Name of dataset. Must be in `datasets.available_datasets`

'''
    if dataset_name not in datasets:
        raise Exception(f"Unknown Dataset: {dataset_name}")


    interim_dataset_path = interim_data_path / dataset_name

    logger.info(f"Checking for {dataset_name}")
    if datasets[dataset_name].get('url_list', None):
        single_file = False
        status, results = fetch_files(dst_dir=raw_data_path,
                                      **datasets[dataset_name])
        if status:
            logger.info(f"Retrieved Dataset successfully")
        else:
            logger.error(f"Failed to retrieve all data files: {results}")
            raise Exception("Failed to retrieve all data files")
        if do_unpack:
            for _, filename, _ in results:
                unpack(filename, dst_dir=interim_dataset_path)
    else:
        single_file = True
        status, filename, hashval = fetch_file(dst_dir=raw_data_path,
                                               **datasets[dataset_name])
        hashtype = datasets[dataset_name].get('hash_type', None)
        if status:
            logger.info(f"Retrieved Dataset: {dataset_name} "
                        f"({hashtype}: {hashval})")
        else:
            logger.error(f"Unpack to {filename} failed (hash: {hashval}). "
                         f"Status: {status}")
            raise Exception(f"Failed to download raw data: {filename}")
        if do_unpack:
            unpack(filename, dst_dir=interim_dataset_path)
    if do_unpack:
        return interim_dataset_path
    else:
        if single_file:
            return filename
        else:
            return raw_data_path

def load_coil_20():
    c20 = Bunch()
    feature_vectors = []
    glob_path = paths.interim_data_path / 'coil-20' / 'processed_images' / '*.pgm'
    filelist = glob.glob(str(glob_path))
    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    c20['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    c20['data'] = np.vstack(feature_vectors)
    with open(_MODULE_DIR / 'coil-20.txt') as fd:
        c20['DESCR'] = fd.read()
    return c20

def load_coil_100():
    c100 = Bunch()
    feature_vectors = []
    glob_path = paths.interim_data_path / 'coil-100' / 'coil-100/' / '*.ppm'
    filelist = glob.glob(str(glob_path))
    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    c100['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    c100['data'] = np.vstack(feature_vectors)
    with open(_MODULE_DIR / 'coil-100.txt') as fd:
        c100['DESCR'] = fd.read()
    return c100

def load_fmnist(kind='train', return_X_y=False):
    '''
    Load the fashion-MNIST dataset
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''
    fmnist = Bunch()

    if kind == 'test':
        kind = 't10k'

    label_path = paths.interim_data_path / 'f-mnist' / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        fmnist['target'] = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    data_path = paths.interim_data_path / 'f-mnist' / f"{kind}-images-idx3-ubyte"
    with open(data_path, 'rb') as fd:
        fmnist['data'] = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(fmnist['target']), 784)
    with open(_MODULE_DIR / 'f-mnist.txt') as fd:
        fmnist['DESCR'] = fd.read()

    if return_X_y:
        return fmnist.data, fmnist.target
    else:
        return fmnist

def load_mnist(kind='train', variant='mnist', return_X_y=False):
    '''
    Load the MNIST dataset
    variant: {'mnist', 'f-mnist'}
        Which variant to load
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''
    dset = Bunch()

    if kind == 'test':
        kind = 't10k'

    label_path = paths.interim_data_path / variant / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        dset['target'] = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    data_path = paths.interim_data_path / variant / f"{kind}-images-idx3-ubyte"
    with open(data_path, 'rb') as fd:
        dset['data'] = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(dset['target']), 784)
    with open(_MODULE_DIR / f'{variant}.txt') as fd:
        dset['DESCR'] = fd.read()

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset



def load_dataset(dataset_name, return_X_y=False, **kwargs):
    '''Loads a scikit-learn style dataset

    dataset_name:
        Name of dataset to load
    return_X_y: boolean, default=False
        if True, returns (data, target) instead of a Bunch object
    '''

    if dataset_name not in datasets:
        raise Exception(f'Unknown Dataset: {dataset_name}')

    dset = datasets[dataset_name]['load_function'](**kwargs)

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset

def load_frey_faces(return_X_y=False):
    '''
    Load the Frey Faces dataset

    Note, there are no labels associated with this dataset; i.e.
    `target` is a vector of all zeros
    '''
    frey_file = fetch('frey-faces')

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
    if path is None:
        path = _MODULE_DIR
    else:
        path = pathlib.Path(path)

    ds = datasets.copy()
    # copy, adjusting non-serializable items
    for key, entry in ds.items():
        func = entry['load_function']
        del(entry['load_function'])
        entry['load_function_name'] = func.func.__name__
        entry['load_function_options'] = func.keywords
    print(ds)
    with open(path / filename, 'w') as fw:
        json.dump(ds, fw, indent=indent, sort_keys=sort_keys)

def read_dataset(path=None, filename="datasets.json"):
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
        func = getattr(_MODULE, dset_opts['load_function_name'], fail_func)
        dset_opts['load_function'] = partial(func, **opts)

    return ds

def unknown_function(args, **kwargs):
    """Placeholder for unknown function_name"""
    raise Exception("Unknown function: {args}, {kwargs}")

datasets = read_dataset()

available_datasets = tuple(datasets.keys())
