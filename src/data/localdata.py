import cv2
from ..paths import interim_data_path, processed_data_path
import glob
import pandas as pd
import numpy as np
import os
import logging
from scipy.io import loadmat

from .utils import read_space_delimited, normalize_labels

__all__ = ['process_coil', 'process_frey_faces', 'process_hiva',
           'process_lvq_pak', 'process_mnist', 'process_orl_faces', 'process_shuttle_statlog']

logger = logging.getLogger(__name__)

def process_coil(dataset_name='coil-20', metadata=None, preview_extension=None,
              unpacked_path='processed_images', image_glob='*.pgm', colorspace='greyscale'):
    """Load a coil-style (image) dataset

    dataset_name: string
    metadata: dict
        new metadata will be appended to this dict. If None, a new dict is created
    unpacked_path: string
        Look for data in `interim_data_path/dataset_name/unpacked_path`
    image_glob: glob
        If `preview_extension` is not None, input images are expected
        to match this pattern.
    colorspace: {'greyscale', 'color'},
        Colorspace to use when loading.
    preview_extension: string or None
        Create preview images (of this file type) in
        `processed_data_path / dataset_name / preview_extension`
        Must be a filename extension supported by the image library.
        If None, no preview images are created
    Additional metadata:
        filename: original filename
        rotation: rotation of target (extracted from filename)
    """

    if colorspace is not None:
        if colorspace == 'greyscale':
            colorspace = cv2.COLORSPACE_GRAY
        elif colorspace == 'color':
            colorspace = None
        else:
            raise Exception(f"Unknown colorspace: {colorspace}")

    if metadata is None:
        metadata = {}
    feature_vectors = []
    glob_path = interim_data_path / dataset_name / unpacked_path

    if preview_extension is not None:
        logger.debug(f"creating {preview_extension}-format preview images")
        preview_dir = processed_data_path / dataset_name / preview_extension
        if not preview_dir.exists():
            os.makedirs(preview_dir)

    filename_list = []
    logger.debug(f"Processing images in {unpacked_path} matching {image_glob}")
    for filename in glob_path.glob(image_glob):
        filename_list.append(filename.name)
        if colorspace is not None:
            im = cv2.imread(str(filename), colorspace)
        else:
            im = cv2.imread(str(filename))
        feature_vectors.append(im.flatten())
        if preview_extension is not None:
            preview_file = preview_dir / f"{filename.stem}.{preview_extension}"
            cv2.imwrite(str(preview_file), im)

    metadata['filename'] = pd.Series(filename_list)
    metadata['rotation'] = metadata['filename'].str.extract("obj[0-9]+__([0-9]+)", expand=False)
    target = metadata['filename'].str.extract("obj([0-9]+)", expand=False)
    data = np.vstack(feature_vectors)
    logger.debug(f"Processed {len(feature_vectors)} images")
    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }

    return dset_opts

def process_mnist(dataset_name='mnist', kind='train', metadata=None):
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
        'metadata': metadata,
    }
    return dset_opts

def process_orl_faces(dataset_name='orl-faces', metadata=None):
    """Load the ORL Faces dataset

        Consists of 92x112, 8-bit greyscale images of 40 total subjects
    """
    extract_dir = interim_data_path / dataset_name

    if metadata is None:
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
    metadata['filename'] = np.array(filename)

    dset_opts = {
        'dataset_name': dataset_name,
        'data': data,
        'target': target,
        'metadata': metadata
    }
    return dset_opts

def process_hiva(dataset_name='hiva', kind='train', metadata=None):
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
        'metadata': metadata,
    }
    return dset_opts

def process_frey_faces(dataset_name='frey-faces', filename='frey_rawface.mat', metadata=None):
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
        'metadata': metadata
    }
    return dset_opts

def process_lvq_pak(dataset_name='lvq-pak', kind='all', numeric_labels=True, metadata=None):
    """
    kind: {'test', 'train', 'all'}, default 'all'
    numeric_labels: boolean (default: True)
        if set, target is a vector of integers, and label_map is created in the metadata
        to reflect the mapping to the string targets
    """

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
        if metadata is None:
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

def process_shuttle_statlog(dataset_name='shuttle-statlog', kind='train', metadata=None):
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
        'metadata': metadata,
    }
    return dset_opts
