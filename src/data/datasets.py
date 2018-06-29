import hashlib
import os
import pathlib
import tarfile
import requests
import logging

from utils import fetch_file, unpack_tgz
logger = logging.getLogger(__name__)


datasets = {
    'coil-20': {
        'url': 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.tar.gz',
        'hash_type': 'sha1',
        'hash_value': 'e5d518fa9ef1d81aef7dfa24b398e4a509b2ffd5',
        },
    'coil-100': {
        'url': 'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.tar.gz',
        'hash_type': 'sha1',
        'hash_value': 'b58920394780e1c224a39004e74bd3574fbed85a',
        },
}

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

