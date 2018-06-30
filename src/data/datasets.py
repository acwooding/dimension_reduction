import hashlib
import os
import pathlib
import tarfile
import requests
import logging
import glob
import cv2
import pandas as pd
import numpy as np
from sklearn.datasets.base import Bunch
from .. import paths

_module_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))

from .utils import fetch_file, unpack_tgz
logger = logging.getLogger(__name__)



def fetch_and_unpack(dataset_name, data_dir=None):
    '''Fetch and unpack a dataset'''
    if dataset_name not in datasets:
        raise Exception(f"Unknown Dataset: {dataset_name}")

    if data_dir is None:
        data_dir = pathlib.Path('.')

    raw_data_dir = data_dir / 'raw'
    interim_data_dir = data_dir / 'interim' / dataset_name

    status, filename, hashval = fetch_file(dst_dir=raw_data_dir,
                                           **datasets[dataset_name])
    if status:
        logger.info(f"Retrieved Dataset: {dataset_name} "
                    f"({datasets[dataset_name]['hash_type']}: {hashval})")
        unpack_tgz(filename, dst_dir=interim_data_dir)
    else:
        logger.error(f"Unpack to {filename} failed (hash: {hashval}). "
                     f"Status: {status}")
        raise Exception(f"Failed to download raw data: {filename}")
    return interim_data_dir

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
    with open(_module_dir / 'coil-20.txt') as fd:
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
    with open(_module_dir / 'coil-100.txt') as fd:
        c100['DESCR'] = fd.read()
    return c100

def load_dataset(dataset_name):
    '''Loads a scikit-learn style dataset'''

    if dataset_name not in _datasets:
        raise Exception(f'Unknown Dataset: {dataset_name}')

    return _datasets[dataset_name]['load_function']() 

_datasets = {
    'coil-20': {
        'url': 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.tar.gz',
        'hash_type': 'sha1',
        'hash_value': 'e5d518fa9ef1d81aef7dfa24b398e4a509b2ffd5',
        'load_function': load_coil_20,
        },
    'coil-100': {
        'url': 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz',
        'hash_type': 'sha1',
        'hash_value': 'b58920394780e1c224a39004e74bd3574fbed85a',
        'load_function': load_coil_100,
        },
}

available_datasets = tuple(_datasets.keys())


