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
from src import paths
from .utils import fetch_file, unpack, fetch_files


_module_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)


def fetch_and_unpack(dataset_name, data_dir=None):
    '''Fetch and unpack a dataset'''
    if dataset_name not in _datasets:
        raise Exception(f"Unknown Dataset: {dataset_name}")

    if data_dir is None:
        data_dir = pathlib.Path('.')

    raw_data_dir = data_dir / 'raw'
    interim_dataset_dir = data_dir / 'interim' / dataset_name
    logger.info(f"Checking for {dataset_name}")
    if _datasets[dataset_name].get('url_list', None):
        status, results = fetch_files(dst_dir=raw_data_dir,
                                      **_datasets[dataset_name])
        if status:
            logger.info(f"Retrieved Dataset successfully")
        else:
            logger.error(f"Failed to retrieve all data files: {results}")
            raise Exception("Failed to retrieve all data files")
        for _, filename, _ in results:
            logger.info(f"Unpacking {dataset_name}")
            unpack(filename, dst_dir=interim_dataset_dir)
    else:
        status, filename, hashval = fetch_file(dst_dir=raw_data_dir,
                                               **_datasets[dataset_name])
        if status:
            logger.info(f"Retrieved Dataset: {dataset_name} "
                        f"({_datasets[dataset_name]['hash_type']}: {hashval})")
        else:
            logger.error(f"Unpack to {filename} failed (hash: {hashval}). "
                         f"Status: {status}")
            raise Exception(f"Failed to download raw data: {filename}")
        logger.info(f"Unpacking {dataset_name}")
        unpack(filename, dst_dir=interim_dataset_dir)

    return interim_dataset_dir

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

def load_fmnist():
    fmnist = Bunch()
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
    'f-mnist': {
        'url_list': [
            {
                'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                'hash_type': 'md5',
                'hash_value': '8d4fb7e6c68d591d4c3dfef9ec88bf0d',
                'name': 'training_data',
            },
            {
                'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                'hash_type': 'md5',
                'hash_value': '25c81989df183df01b3e8a0aad5dffbe',
                'name': 'training_labels',
            },
            {
                'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                'hash_type': 'md5',
                'hash_value': 'bef4ecab320f06d8554ea6380940ec79',
                'name': 'test_data'
            },
            {
                'url': 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
                'hash_type': 'md5',
                'hash_value': 'bb300cfdad3c16e7a12a480ee83cd310',
                'name': 'test_labels'
            },
        ],
        'load_function': load_fmnist,
    },
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
