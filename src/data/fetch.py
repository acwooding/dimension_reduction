import hashlib
import gzip
import os
import pathlib
import shutil
import tarfile
import zipfile
import zlib
import requests

from ..paths import raw_data_path, interim_data_path
from ..logging import logger

__all__ = [
    'available_hashes',
    'fetch_file',
    'fetch_files',
    'fetch_text_file',
    'hash_file',
    'unpack'
]

_HASH_FUNCTION_MAP = {
    'md5': hashlib.md5,
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
}

def available_hashes():
    """Valid Hash Functions

    This function simply returns the dict known hash function
    algorithms.

    It exists to allow for a description of the mapping for
    each of the valid strings.

    The hash functions are:

    ============     ====================================
    Algorithm        Function
    ============     ====================================
    md5              hashlib.md5
    sha1             hashlib.sha1
    sha256           hashlib.sha256
    ============     ====================================

    >>> list(available_hashes().keys())
    ['md5', 'sha1', 'sha256']
    """
    return _HASH_FUNCTION_MAP

def hash_file(fname, algorithm="sha1", block_size=4096):
    '''Compute the hash of an on-disk file

    algorithm: {'md5', sha1', 'sha256'}
        hash algorithm to use
    block_size:
        size of chunks to read when hashing

    Returns:
        Hashlib object
    '''
    hashval = _HASH_FUNCTION_MAP[algorithm]()
    with open(fname, "rb") as fd:
        for chunk in iter(lambda: fd.read(block_size), b""):
            hashval.update(chunk)
    return hashval

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

    Examples
    --------
    >>> fetch_files()
    Traceback (most recent call last):
      ...
    Exception: One of `file_name` or `url` is required
    '''
    url_list = kwargs.get('url_list', None)
    if not url_list:
        return fetch_file(force=force, dst_dir=dst_dir, **kwargs)
    result_list = []
    for url_dict in url_list:
        name = url_dict.get('name', None)
        if name is None:
            name = url_dict.get('url', 'dataset')
        logger.debug(f"Ready to fetch {name}")
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
        _, filename, _ = retlist
        with open(filename, 'r') as txt:
            return txt.read()
    else:
        logger.warning(f'fetch of {url} failed with status: {retlist[0]}')
        return None

def fetch_file(url=None, contents=None,
               file_name=None, dst_dir=None,
               force=False,
               hash_type="sha1", hash_value=None,
               **kwargs):
    '''Fetch remote files via URL

    if `file_name` already exists, compute the hash of the on-disk file

    contents:
        contents of file to be created
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

    Returns
    -------
    one of:
        (HTTP_Code, downloaded_filename, hash) (if downloaded from URL)
        (True, filename, hash) (if already exists)
        (False, [error])

    Examples
    --------
    >>> fetch_file()
    Traceback (most recent call last):
      ...
    Exception: One of `file_name` or `url` is required
    '''
    if dst_dir is None:
        dst_dir = raw_data_path
    if file_name is None:
        if url is None:
            raise Exception('One of `file_name` or `url` is required')
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
                    logger.debug(f"{file_name} already exists and hash is valid")
                    return True, raw_data_file, raw_file_hash
            else:
                logger.warning(f"{file_name} exists but has bad hash {raw_file_hash}."
                               " Re-downloading")
        else:
            if force is False:
                logger.debug(f"{file_name} exists, but no hash to check")
                return True, raw_data_file, raw_file_hash

    if url is None and contents is None:
        raise Exception("One of `url` or `contents` must be specified if `file_name` doesn't yet exist")

    if url is not None:
        # Download the file
        try:
            results = requests.get(url)
            results.raise_for_status()
            raw_file_hash = _HASH_FUNCTION_MAP[hash_type](results.content).hexdigest()
            if hash_value is not None:
                if raw_file_hash != hash_value:
                    print(f"Invalid hash on downloaded {file_name}"
                          f" ({hash_type}:{raw_file_hash}) != {hash_type}:{hash_value}")
                    return False, None, raw_file_hash
            logger.debug(f"Writing {raw_data_file}")
            with open(raw_data_file, "wb") as code:
                code.write(results.content)
        except requests.exceptions.HTTPError as err:
            return False, err, None
    elif contents is not None:
        with open(raw_data_file, 'w') as fw:
            fw.write(contents)
        raw_file_hash = hash_file(raw_data_file, algorithm=hash_type).hexdigest()
        return True, raw_data_file, raw_file_hash
    else:
        raise Exception('One of `url` or `contents` must be specified')

    return results.status_code, raw_data_file, raw_file_hash

def unpack(filename, dst_dir=None, create_dst=True):
    '''Unpack a compressed file

    filename: path
        file to unpack
    dst_dir: path (default paths.interim_data_path)
        destination directory for the unpack
    create_dst: boolean
        create the destination directory if needed
    '''
    if dst_dir is None:
        dst_dir = interim_data_path

    if create_dst:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    # in case it is a Path
    path = str(filename)

    archive = False
    if path.endswith('.zip'):
        archive = True
        opener, mode = zipfile.ZipFile, 'r'
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        archive = True
        opener, mode = tarfile.open, 'r:gz'
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        archive = True
        opener, mode = tarfile.open, 'r:bz2'
    elif path.endswith('.tar'):
        archive = True
        opener, mode = tarfile.open, 'r'
    elif path.endswith('.gz'):
        opener, mode = gzip.open, 'rb'
        outfile, outmode = path[:-3], 'wb'
    elif path.endswith('.Z'):
        logger.warning(".Z files are only supported on systems that ship with gzip. Trying...")
        os.system(f'gzip -f -d {path}')
        opener, mode = open, 'rb'
        path = path[:-2]
        outfile, outmode = path, 'wb'
    else:
        opener, mode = open, 'rb'
        outfile, outmode = path, 'wb'
        logger.info("No compression detected. Copying...")

    with opener(path, mode) as f_in:
        if archive:
            logger.debug(f"Extracting {filename.name}")
            f_in.extractall(path=dst_dir)
        else:
            outfile = pathlib.Path(outfile).name
            logger.info(f"Decompresing {outfile}")
            with open(pathlib.Path(dst_dir) / outfile, outmode) as f_out:
                shutil.copyfileobj(f_in, f_out)
