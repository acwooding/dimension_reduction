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

from .utils import fetch_file, unpack_tgz
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



def load_coil_100(dataset_dir):
    c100 = Bunch()
    feature_vectors = []
    glob_path = pathlib.Path(dataset_dir) / 'coil-100' / '*.ppm'
    filelist = glob.glob(str(glob_path))
    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    c100['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    c100['data'] = np.vstack(feature_vectors)
    c100['DESCR'] = '''
Columbia University Image Library
=================================

Notes
-----
Data Set Characteristics:
    :Number of Instances: 7200
    :Number of Attributes: 49152
    :Attribute Information: 128x128 image of 3 channels (16-bit RGB values)
    :Missing Attribute Values: None
    :Creator: Sameer A Nene, Shree K. Nayar and Hiroshi Murase
    :Date: 1995

This is a copy of the Columbia Object Image Library (COIL-100) data:
http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip

Columbia Object Image Library (COIL-100) is a database of color images of 100 objects.

This dataset consists of 7,200 color images of 100 objects (72 images per object) where the objects
have been placed on a motorized turntable against a black background. Images were taken at 5 degree rotations,
giving 72 images per object. The resulting images were then size and intensity-normalized.

Size normalization involved first clipping to a rectangual bounding box and resized (with aspect
ratio preserved) to 128x128 using interpolation-decimation filters to minimize aliasing [Oppenheim and Schafer-1989].

To normalize intensity, every image was histogram stretched, i.e. the intensity
of the brightest pixel was made 65535 and intensities of the other pixels were scaled accordingly.
The images were saved as 16-bit PPM (portable pixmap) color images.

Raw data files are named according to object and rotation; e.g.  `obj7__10.ppm`.
The prefix "obj7" identifies the object, and "10" indicates the rotation in degrees.

A greyscale version of this dataset is also available (COIL-20)

References
----------
  - [Murase and Nayar, 1995] H. Murase and S. K. Nayar. Visual Learning and Recognition
of 3D Objects from Appearance. International Journal of Computer Vision , 14(1):5{24,
January 1995.
  - [Nayar and Poggio, 1996] S. K. Nayar and T. Poggio. Early Visual Learning. In S. K. Nayar
and T. Poggio, editors, Early Visual Learning . Oxford University Press, March 1996.
  - [Nayar et al. , 1996a] S. K. Nayar, H. Murase and S. A. Nene. Parametric Appearance Repre-
sentation. In S. K. Nayar and T. Poggio, editors, Early Visual Learning . Oxford University
Press, March 1996.
  - [Nayar et al. , 1996b] S. K. Nayar, S. A. Nene and H. Murase. Real-Time 100 Object Recog-
nition System. In Proceedings of ARPA Image Understanding Workshop , Palm Springs,
February 1996.
  - [Nayar et al. , 1996c] S. K. Nayar, S. A. Nene and H. Murase. Real-Time 100 Object Recog-
nition System. In Proceedings of IEEE International Conference on Robotics and Automa-
tion , Minneapolis, April 1996.
  - [Nene and Nayar, 1994] S. A. Nene and S. K. Nayar. SLAM: A Software Library for Ap-
pearance Matching. In Proceedings of ARPA Image Understanding Workshop , Monterey,
November 1994. Also Technical Report CUCS-019-94.
  - [Nene et al. , 1996] S. A. Nene, S. K. Nayar and H. Murase. Columbia Object Image Library:
COIL-20. Technical Report CUCS-005-96, Department of Computer Science, Columbia
University, February 1996.
  - [Oppenheim and Schafer, 1989] A. V. Oppenheim and R. W. Schafer. Discrete-Time Signal
Processing , chapter 3, pages 111{130. Prentice Hall, 1989.

'''
    return c100


def load_dataset(dataset_name, data_dir=None):
    '''Loads a scikit-learn style dataset'''

    interim_path = pathlib.Path(data_dir) / dataset_name
    if dataset_name == 'coil-100':
        dataset = load_coil_100(interim_path)
    else:
        raise Exception(f'Unknown Dataset: {dataset_name}')

    return dataset
