import cv2
from ..paths import interim_data_path
import glob
import pandas as pd
import numpy as np
import os
from scipy.io import loadmat

from .utils import read_space_delimited, normalize_labels

__all__ = ['load_coil_20', 'load_coil_100', 'load_frey_faces', 'load_hiva',
           'load_lvq_pak', 'load_mnist', 'load_orl_faces', 'load_shuttle_statlog']

def load_coil_20(dataset_name='coil-20', metadata=None):
    """ Load the coil 20 dataset

    Additional metadata:
        filename: original filename
        rotation: rotation of target (extracted from filename)
    """
    if metadata is None:
        metadata = {}
    feature_vectors = []
    glob_path = interim_data_path / 'coil-20' / 'processed_images' / '*.pgm'
    filelist = glob.glob(str(glob_path))

    metadata['filename'] = pd.Series(filelist).apply(os.path.basename)
    metadata['rotation'] = pd.Series(filelist).str.extract("obj[0-9]+__([0-9]+)", expand=False)

    for filename in filelist:
        im = cv2.imread(filename, cv2.COLORSPACE_GRAY)
        feature_vectors.append(im.flatten())

    target = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    data = np.vstack(feature_vectors)

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }

    return dset_opts

def load_coil_100(dataset_name='coil-100'):
    """ Load the coil 100 dataset

    Additional metadata:
        filename: original filename
        rotation: rotation of target (extracted from filename)
    """

    metadata = {}
    feature_vectors = []
    glob_path = interim_data_path / 'coil-100' / 'coil-100/' / '*.ppm'
    filelist = glob.glob(str(glob_path))

    metadata['filename'] = pd.Series(filelist).apply(os.path.basename)
    metadata['rotation'] = pd.Series(filelist).str.extract("obj[0-9]+__([0-9]+)", expand=False)

    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    target = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    data = np.vstack(feature_vectors)

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }

    return dset_opts

def load_mnist(dataset_name='mnist', kind='train'):
    '''
    Load the MNIST dataset (or a compatible variant; e.g. F-MNIST)

    dataset_name: {'mnist', 'f-mnist'}
        Which variant to load
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''
    if kind == 'test':
        kind = 't10k'

    label_path = interim_data_path / dataset_name / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        target = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    dataset_path = interim_data_path / dataset_name / f"{kind}-images-idx3-ubyte"
    with open(dataset_path, 'rb') as fd:
        data = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(target), 784)
    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
    }
    return dset_opts

def load_orl_faces(dataset_name='orl-faces'):
    """Load the ORL Faces dataset

        Consists of 92x112, 8-bit greyscale images of 40 total subjects
    """
    extract_dir = interim_data_path / dataset_name

    metadata = {}
    filename = []
    target = []
    feature_vectors = []
    for subject_dir in extract_dir.iterdir():
        if subject_dir.is_dir():
            subject = subject_dir.name[1:]
            for file in subject_dir.iterdir():
                filename.append(file.name)
                target.append(subject)
                im = cv2.imread(str(file), cv2.COLORSPACE_GRAY)
                feature_vectors.append(im.flatten())
    target = np.array(target)
    data = np.vstack(feature_vectors)
    metadata['filenames'] = np.array(filename)

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }
    return dset_opts

def load_hiva(dataset_name='hiva', kind='train'):
    """Load the HIVA dataset

    kind: {'train', 'test', 'valid'}
        if 'test' or 'valid', empty labels will be generated.
        Labels are generated only if 'train' is specified
    """
    if kind not in ['train', 'test', 'valid']:
        raise Exception(f"Unknown kind: {kind}")

    hiva_dir = interim_data_path / dataset_name / 'HIVA'

    data = np.genfromtxt(hiva_dir / f'hiva_{kind}.data')

    if kind == 'train':
        labels = np.genfromtxt(hiva_dir / f'hiva_{kind}.labels')
    else:
        labels = np.zeros(data.shape[0])

    data = data
    target = labels

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
    }
    return dset_opts

def load_frey_faces(dataset_name='frey-faces', filename='frey_rawface.mat'):
    '''
    Load the Frey Faces dataset

    Note, there are no labels associated with this dataset; i.e.
    `target` is a vector of all zeros
    '''

    frey_file = interim_data_path / dataset_name / filename

    ff = loadmat(frey_file, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T

    data = ff
    target = np.zeros(ff.shape[0])

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
    }
    return dset_opts

def load_lvq_pak(dataset_name='lvq-pak', kind='all', numeric_labels=True):
    """
    kind: {'test', 'train', 'all'}, default 'all'
    numeric_labels: boolean (default: True)
        if set, target is a vector of integers, and label_map is created in the metadata
        to reflect the mapping to the string targets
    """

    metadata = None
    untar_dir = interim_data_path / dataset_name
    unpack_dir = untar_dir / 'lvq_pak-3.1'

    if kind == 'train':
        data, target = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
    elif kind == 'test':
        data, target = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
    elif kind == 'all':
        data1, target1 = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
        data2, target2 = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
        data = np.vstack((data1, data2))
        target = np.append(target1, target2)
    else:
        raise Exception(f'Unknown kind: {kind}')

    if numeric_labels:
        metadata = {}
        mapped_target, label_map = normalize_labels(target)
        metadata['label_map'] = label_map
        target = mapped_target

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }
    return dset_opts

def load_shuttle_statlog(dataset_name='shuttle-statlog', kind='train'):
    """Load the shuttle dataset

    This is a 9-dimensional dataset with class labels split into training and test sets

    kind: {'train', 'test'}
    """
    filename_map = {
        'train': f'shuttle.trn',
        'test': f'shuttle.tst',
    }

    extract_dir = interim_data_path / dataset_name

    data, target = read_space_delimited(extract_dir / filename_map[kind])

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
    }
    return dset_opts
