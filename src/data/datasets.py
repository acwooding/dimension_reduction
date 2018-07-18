from functools import partial
import glob
import joblib
import json
import logging
import os
import pathlib
import requests
import pandas as pd
import numpy as np
from scipy.io import loadmat
from sklearn.datasets.base import Bunch
import sys

from .utils import hash_file, unpack, hash_function_map
from ..paths import data_path, raw_data_path, interim_data_path, processed_data_path

_MODULE = sys.modules[__name__]
_MODULE_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__)))
logger = logging.getLogger(__name__)


def new_dataset(*, dataset_name):
    """Return an unpopulated dataset object.

    Fills in LICENSE and DESCR if they are present.
    Takes metadata from the url_list object if present. Otherwise, if
    `*.license` or `*.readme` files are present in the module directory,
    these will be as LICENSE and DESCR respectively.
    """
    global available_datasets

    dset = Bunch()
    dset['metadata'] = {}
    dset['LICENSE'] = None
    dset['DESCR'] = None
    filemap = {
        'LICENSE': f'{dataset_name}.license',
        'DESCR': f'{dataset_name}.readme'
    }

    # read metadata from disk if present
    for metadata_type in filemap:
        metadata_file = _MODULE_DIR / filemap[metadata_type]
        if metadata_file.exists():
            with open(metadata_file, 'r') as fd:
                dset[metadata_type] = fd.read()

    # Use downloaded metadata if available
    ds = available_datasets[dataset_name]
    for fetch_dict in ds.get('url_list', []):
        name = fetch_dict.get('name', None)
        # if metadata is present in the URL list, use it
        if name in ['DESCR', 'LICENSE']:
            txtfile = get_dataset_filename(fetch_dict)
            with open(raw_data_path / txtfile, 'r') as fr:
                dset[name] = fr.read()

    return dset

def get_dataset_filename(ds_dict):
    '''Figure out the downloaded filename for a dataset entry

    if a `file_name` key is present, use this,
    otherwise, use the last component of the `url`
    '''

    file_name = ds_dict.get('file_name', None)
    url = ds_dict.get('url', [])
    if file_name is None:
        file_name = url.split("/")[-1]
    return file_name

def fetch_files(force=False, dst_dir=None, **kwargs):
    '''
    fetches a list of files via URL

    url_list: list of dicts, each containing:
        url:
            url to be downloaded
        hash_type:
            Type of hash to compute
        hash_value: (optional)
            if specified, the hash of the downloaded file will be
            checked against this value
        name: (optional)
            Name of this dataset component
        raw_file:
            output file name. If not specified, use the last
            component of the URL
    '''
    url_list = kwargs.get('url_list', None)
    if not url_list:
        return fetch_file(force=force, dst_dir=dst_dir, **kwargs)
    result_list = []
    for url_dict in url_list:
        name = url_dict.get('name', 'dataset')
        logger.info(f"Fetching {name}")
        result_list.append(fetch_file(force=force, dst_dir=dst_dir, **url_dict))
    return all([r[0] for r in result_list]), result_list

def fetch_text_file(url, file_name=None, dst_dir=None, force=True, **kwargs):
    """Fetch a text file (via URL) and return it as a string.

    Arguments
    ---------

    file_name:
        output file name. If not specified, use the last
        component of the URL
    dst_dir:
        directory to place downloaded files
    force: boolean
        normally, the URL is only downloaded if `file_name` is
        not present on the filesystem, or if the existing file has a
        bad hash. If force is True, download is always attempted.

    In addition to these options, any of `fetch_file`'s keywords may
    also be passed

    Returns
    -------
    fetched string, or None if something went wrong with the download
    """
    retlist = fetch_file(url, file_name=file_name, dst_dir=dst_dir,
                         force=force, **kwargs)
    if retlist[0]:
        status, filename, hashval = retlist
        with open(filename, 'r') as txt:
            return txt.read()
    else:
        logger.warning(f'fetch of {url} failed with status: {retlist[0]}')
        return None


def fetch_file(url,
               file_name=None, dst_dir=None,
               force=False,
               hash_type="sha1", hash_value=None,
               **kwargs):
    '''Fetch remote files via URL

    url:
        url to be downloaded
    hash_type:
        Type of hash to compute
    hash_value: (optional)
        if specified, the hash of the downloaded file will be
        checked against this value
    name: (optional)
        Name of this dataset component
    file_name:
        output file name. If not specified, use the last
        component of the URL
    dst_dir:
        directory to place downloaded files
    force: boolean
        normally, the URL is only downloaded if `file_name` is
        not present on the filesystem, or if the existing file has a
        bad hash. If force is True, download is always attempted.


    returns one of:


        (HTTP_Code, downloaded_filename, hash) (if downloaded from URL)
        (True, filename, hash) (if already exists)
        (False, [error])
    if `file_name` already exists, compute the hash of the on-disk file,
    '''
    if dst_dir is None:
        dst_dir = raw_data_path
    if file_name is None:
        file_name = url.split("/")[-1]
    dl_data_path = pathlib.Path(dst_dir)

    if not os.path.exists(dl_data_path):
        os.makedirs(dl_data_path)

    raw_data_file = dl_data_path / file_name

    if raw_data_file.exists():
        raw_file_hash = hash_file(raw_data_file, algorithm=hash_type).hexdigest()
        if hash_value is not None:
            if raw_file_hash == hash_value:
                if force is False:
                    logger.info(f"{file_name} exists and hash is valid")
                    return True, raw_data_file, raw_file_hash
            else:
                logger.warning(f"{file_name} exists but has bad hash {raw_file_hash}."
                               " Re-downloading")
        else:
            if force is False:
                logger.info(f"{file_name} exists, but no hash to check")
                return True, raw_data_file, raw_file_hash

    # Download the file
    try:
        results = requests.get(url)
        results.raise_for_status()
        raw_file_hash = hash_function_map[hash_type](results.content).hexdigest()
        if hash_value is not None:
            if raw_file_hash != hash_value:
                print(f"Invalid hash on downloaded {file_name}"
                      f" ({hash_type}:{raw_file_hash}) != {hash_type}:{hash_value}")
                return False, None, raw_file_hash
        logger.info(f"Writing {raw_data_file}")
        with open(raw_data_file, "wb") as code:
            code.write(results.content)
    except requests.exceptions.HTTPError as err:
        return False, err, None

    return results.status_code, raw_data_file, raw_file_hash

def build_dataset_dict(hash_type='sha1', hash_value=None, url=None,
                       name=None, file_name=None):
    """fetch a URL, return a dataset dictionary entry

    hash_type: {'sha1', 'md5', 'sha256'}
    hash_value: string or None
        if None, hash will be computed from downloaded file
    file_name: string or None
        Name of downloaded file. If None, will be the last component of the URL
    url: URL to fetch

    returns: dict
    """
    fetch_dict = {'url': url, 'hash_type':hash_type, 'hash_value':hash_value, 'name': name, 'file_name':file_name}
    status, path, hash_value = fetch_files(**fetch_dict)
    if status:
        fetch_dict['hash_value'] = hash_value
        return fetch_dict

    raise Exception(f"fetch of {url} returned status: {status}")

def fetch_and_unpack(dataset_name, do_unpack=True):
    '''Fetch and process datasets to their usable form

    dataset_name: string
        Name of dataset. Must be in `datasets.json`
    do_unpack: boolean
        If false, just download, don't process.

    '''
    ds = read_datasets()
    if dataset_name not in ds:
        raise Exception(f"Unknown Dataset: {dataset_name}")

    interim_dataset_path = interim_data_path / dataset_name

    logger.info(f"Checking for {dataset_name}")
    if ds[dataset_name].get('url_list', None):
        single_file = False
        status, results = fetch_files(dst_dir=raw_data_path,
                                      **ds[dataset_name])
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
                                               **ds[dataset_name])
        hashtype = ds[dataset_name].get('hash_type', None)
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

def read_space_delimited(filename, skiprows=None, class_labels=True):
    """Read an space-delimited file

    skiprows: list of rows to skip when reading the file.

    Note: we can't use automatic comment detection, as
    `#` characters are also used as data labels.
    class_labels: boolean
        if true, the last column is treated as the class label
    """
    with open(filename, 'r') as fd:
        df = pd.read_table(fd, skiprows=skiprows, skip_blank_lines=True, comment=None, header=None, sep=' ', dtype=str)
        # targets are last column. Data is everything else
        if class_labels is True:
            target = df.loc[:,df.columns[-1]].values
            data = df.loc[:,df.columns[:-1]].values
        else:
            data = df.values
            target = np.zeros(data.shape[0])
        return data, target

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
    raise Exception(f"Unknown function: {args}, {kwargs}")

def write_datasets(path=None, filename="datasets.json", indent=4, sort_keys=True):
    """Write a serialized (JSON) dataset file"""
    if path is None:
        path = _MODULE_DIR
    else:
        path = pathlib.Path(path)

    ds = available_datasets.copy()
    # copy, adjusting non-serializable items
    for key, entry in ds.items():
        func = entry.get('load_function', None)
        if func is None:
             func = partial(new_dataset, dataset_name=key)
        else:
            del(entry['load_function'])
        entry['load_function_name'] = func.func.__name__
        entry['load_function_options'] = func.keywords
    with open(path / filename, 'w') as fw:
        json.dump(ds, fw, indent=indent, sort_keys=sort_keys)

def sync_datasets():
    """Make sure the internal and on-disk dataset lists are in sync"""
    global available_datasets

    write_datasets()
    available_datasets = read_datasets()

def add_dataset_by_urllist(dataset_name, url_list):
    """Add a new dataset by specifying a url_list

    url_list is a list of dicts keyed by:
        * url, hash_type, hash_value, name, file_name
    """
    global available_datasets

    available_datasets[dataset_name] = {'url_list': url_list}
    sync_datasets()
    return available_datasets[dataset_name]

def add_dataset_metadata(dataset_name, from_file=None, from_str=None, kind='DESCR'):
    """Add metadata to a dataset

    from_file: create metadata entry from contents of this file
    from_str: create metadata entry from this string
    kind: {'DESCR', 'LICENSE'}
    """
    global available_datasets

    filename_map = {
        'DESCR': f'{dataset_name}.readme',
        'LICENSE': f'{dataset_name}.license',
    }

    if dataset_name not in available_datasets:
        raise Exception(f'No such dataset: {dataset_name}')

    if kind not in filename_map:
        raise Exception(f'Unknown kind: {kind}. Must be one of {filename_map.keys()}')

    if from_file is not None:
        with open(from_file, 'r') as fd:
            meta_txt = fd.read()
    elif from_str is not None:
        meta_txt = from_str
    else:
        raise Exception(f'One of `from_file` or `from_str` is required')

    with open(_MODULE_DIR / filename_map[kind], 'w') as fw:
        fw.write(meta_txt)

def add_dataset_function(dataset_name, function):
    """Add a load function for the given dataset_name"""
    global available_datasets

    available_datasets[dataset_name]['load_function'] = partial(function)
    sync_datasets()
    return available_datasets[dataset_name]

def load_dataset(dataset_name, return_X_y=False, force=False, **kwargs):
    '''Loads a scikit-learn style dataset

    dataset_name:
        Name of dataset to load
    return_X_y: boolean, default=False
        if True, returns (data, target) instead of a Bunch object
    force: boolean
        if True, do complete fetch/process cycle. If False, will use cached object (if present)
    '''

    if dataset_name not in available_datasets:
        raise Exception(f'Unknown Dataset: {dataset_name}')

    # check for cached version
    cache_file = processed_data_path / f'{dataset_name}.jlib'
    if cache_file.exists() and force is not True:
        dset = joblib.load(cache_file)
    else:
        # no cache. Regenerate
        fetch_and_unpack(dataset_name)
        dset = available_datasets[dataset_name]['load_function'](**kwargs)
        os.makedirs(cache_file.parent, exist_ok=True)
        with open(cache_file, 'wb') as fo:
            joblib.dump(dset, fo)

    if return_X_y:
        return dset.data, dset.target
    else:
        return dset

def labels_to_int(target):
    """Map an arbitary target vector to an integer vector

    Returns
    -------
    tuple: (mapped_target, label_map)

    where:
        mapped_target: integer vector of same shape as target
        label_map: dict mapping integers to original labels such that
            `all(np.vectorize(label_map.get)(mapped_target) == target) == True`
    """
    label_map = {k:v for k, v in enumerate(np.unique(target))}
    label_map_inv = {v:k for k, v in label_map.items()}
    mapped_target = np.vectorize(label_map_inv.get)(target)

    return mapped_target, label_map

#############################################
# Add project-specific import functions
#############################################

def load_coil_20(dataset_name='coil-20'):
    """ Load the coil 20 dataset

    Additional metadata:
        filename: original filename
        rotation: rotation of target (extracted from filename)
    """

    dset = new_dataset(dataset_name=dataset_name)

    dset['metadata'] = {}

    feature_vectors = []
    glob_path = interim_data_path / 'coil-20' / 'processed_images' / '*.pgm'
    filelist = glob.glob(str(glob_path))

    dset.metadata['filename'] = pd.Series(filelist).apply(os.path.basename)
    dset.metadata['rotation'] = pd.Series(filelist).str.extract("obj[0-9]+__([0-9]+)", expand=False)

    for i, filename in enumerate(filelist):
        im = cv2.imread(filename, cv2.COLORSPACE_GRAY)
        feature_vectors.append(im.flatten())

    dset['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    dset['data'] = np.vstack(feature_vectors)
    return dset

def load_coil_100(dataset_name='coil-100'):
    """ Load the coil 100 dataset

    Additional metadata:
        filename: original filename
        rotation: rotation of target (extracted from filename)
    """

    dset = new_dataset(dataset_name=dataset_name)

    dset['metadata'] = {}

    feature_vectors = []
    glob_path = interim_data_path / 'coil-100' / 'coil-100/' / '*.ppm'
    filelist = glob.glob(str(glob_path))

    dset.metadata['filename'] = pd.Series(filelist).apply(os.path.basename)
    dset.metadata['rotation'] = pd.Series(filelist).str.extract("obj[0-9]+__([0-9]+)", expand=False)

    for filename in filelist:
        im = cv2.imread(filename)
        feature_vectors.append(im.flatten())

    dset['target'] = pd.Series(filelist).str.extract("obj([0-9]+)", expand=False)
    dset['data'] = np.vstack(feature_vectors)

    return dset

def load_mnist(dataset_name='mnist', kind='train'):
    '''
    Load the MNIST dataset (or a compatible variant; e.g. F-MNIST)

    dataset_name: {'mnist', 'f-mnist'}
        Which variant to load
    kind: {'train', 'test'}
        Dataset comes pre-split into training and test data.
        Indicates which dataset to load

    '''
    dset = new_dataset(dataset_name=dataset_name)

    if kind == 'test':
        kind = 't10k'

    label_path = interim_data_path / dataset_name / f"{kind}-labels-idx1-ubyte"
    with open(label_path, 'rb') as fd:
        dset['target'] = np.frombuffer(fd.read(), dtype=np.uint8, offset=8)
    dataset_path = interim_data_path / dataset_name / f"{kind}-images-idx3-ubyte"
    with open(dataset_path, 'rb') as fd:
        dset['data'] = np.frombuffer(fd.read(), dtype=np.uint8,
                                       offset=16).reshape(len(dset['target']), 784)
    return dset

def load_orl_faces(dataset_name='orl-faces'):
    """Load the ORL Faces dataset

        Consists of 92x112, 8-bit greyscale images of 40 total subjects
    """
    dset = new_dataset(dataset_name=dataset_name)

    dset['metadata'] = {}

    extract_dir = interim_data_path / dataset_name

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
    dset['target'] = np.array(target)
    dset['data'] = np.vstack(feature_vectors)
    dset.metadata['filename'] = np.array(filename)

    return dset

def load_hiva(dataset_name='hiva', kind='train'):
    """Load the HIVA dataset

    kind: {'train', 'test', 'valid'}
        if 'test' or 'valid', empty labels will be generated.
        Labels are generated only if 'train' is specified
    """
    if kind not in ['train', 'test', 'valid']:
        raise Exception(f"Unknown kind: {kind}")

    hiva_dir = interim_data_path / dataset_name / 'HIVA'

    dset = new_dataset(dataset_name=dataset_name)

    data = np.genfromtxt(hiva_dir / f'hiva_{kind}.data')

    if kind == 'train':
        labels = np.genfromtxt(hiva_dir / f'hiva_{kind}.labels')
    else:
        labels = np.zeros(data.shape[0])

    dset['data'] = data
    dset['target'] = labels
    return dset

def load_frey_faces(dataset_name='frey-faces', filename='frey_rawface.mat'):
    '''
    Load the Frey Faces dataset

    Note, there are no labels associated with this dataset; i.e.
    `target` is a vector of all zeros
    '''

    dset = new_dataset(dataset_name=dataset_name)

    frey_file = interim_data_path / dataset_name / filename

    ff = loadmat(frey_file, squeeze_me=True, struct_as_record=False)
    ff = ff["ff"].T

    dset.data = ff
    dset.target = np.zeros(ff.shape[0])

    return dset

def load_lvq_pak(dataset_name='lvq-pak', kind='all', numeric_labels=True):
    """
    kind: {'test', 'train', 'all'}, default 'all'
    numeric_labels: boolean (default: True)
        if set, target is a vector of integers, and label_map is created in the metadata
        to reflect the mapping to the string targets
    """

    untar_dir = interim_data_path / dataset_name

    unpack_dir = untar_dir / 'lvq_pak-3.1'

    dset = new_dataset(dataset_name=dataset_name)

    if kind == 'train':
        dset['data'], dset['target'] = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
    elif kind == 'test':
        dset['data'], dset['target'] = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
    elif kind == 'all':
        data, target = read_space_delimited(unpack_dir / 'ex1.dat', skiprows=[0,1])
        data2, target2 = read_space_delimited(unpack_dir / 'ex2.dat', skiprows=[0])
        dset['data'] = np.vstack((data, data2))
        dset['target'] = np.append(target, target2)
    else:
        raise Exception(f'Unknown kind: {kind}')

    if numeric_labels:
        mapped_target, label_map = labels_to_int(dset.target)
        dset.metadata['label_map'] = label_map
        dset.target = mapped_target

    return dset

def load_shuttle_statlog(dataset_name='shuttle-statlog', kind='train'):
    """Load the shuttle dataset

    This is a 9-dimensional dataset with class labels split into training and test sets

    kind: {'train', 'test'}
    """
    filename_map = {
        'train': f'shuttle.trn',
        'test': f'shuttle.tst',
    }

    dset = new_dataset(dataset_name=dataset_name)

    extract_dir = interim_data_path / dataset_name

    data, target = read_space_delimited(extract_dir / filename_map[kind])

    dset['data'] = data
    dset['target'] = target

    return dset

#############################################
# End project-specific import functions
#############################################

available_datasets = read_datasets()
