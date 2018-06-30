import hashlib
import os
import pathlib
import tarfile
import requests


hash_function = {
    'sha1': hashlib.sha1,
    'sha256': hashlib.sha256,
    'md5': hashlib.sha256
}


def hash_file(fname, algorithm="sha1", block_size=4096):
    '''Compute the hash of an on-disk file

    algorithm: {'md5', sha1', 'sha256'}
        hash algorithm to use
    block_size:
        size of chunks to read when hashing

    Returns:
        Hashlib object
    '''
    hashval = hash_function[algorithm]()
    with open(fname, "rb") as fd:
        for chunk in iter(lambda: fd.read(block_size), b""):
            hashval.update(chunk)
    return hashval

def fetch_file(url,
               raw_file=None, dst_dir=None,
               force=False,
               hash_type="sha1", hash_value=None,
               **kwargs):
    '''Fetch a remote file via URL

    url:
        url to be downloaded
    raw_file:
        output file name. If not specified, use the last
        component of the URL
    dst_dir:
        directory to place downloaded file
    force: boolean
        normally, the URL is only downloaded if `raw_file` is
        not present on the filesystem, or if the existing file has a
        bad hash. If force is True, a download is always attempted
    hash_type:
        Type of hash to compute
    hash_value: (optional)
        if specified, the hash of the downloaded file will be
        checked against this value


    returns one of:


        (HTTP_Code, downloaded_filename, hash) (if downloaded from URL)
        (True, filename, hash) (if already exists)
        (False, [error])
    if `raw_file` already exists, compute the hash of the on-disk file,
    '''
    if dst_dir is None:
        dst_dir = pathlib.Path(".")
    if raw_file is None:
        raw_file = url.split("/")[-1]
    raw_data_path = pathlib.Path(dst_dir)
    
    if not os.path.exists(raw_data_path):
        os.makedirs(raw_data_path)

    raw_data_file = raw_data_path / raw_file

    if os.path.exists(raw_data_file):
        raw_file_hash = hash_file(raw_data_file, algorithm=hash_type).hexdigest()
        if hash_value is not None:
            if raw_file_hash == hash_value:
                if force is False:
                    return True, raw_data_file, raw_file_hash
            else:
                print(f"{raw_file} exists but has bad hash {raw_file_hash}."
                      " Re-downloading")
        else: # file exists but no hash to check
            if force is False:
                return True, raw_data_file, raw_file_hash

    # Download the file
    try:
        results = requests.get(url)
        results.raise_for_status()
        raw_file_hash = hash_function[hash_type](results.content).hexdigest()
        if hash_value is not None:
            if raw_file_hash != hash_value:
                print(f"Invalid hash on downloaded {raw_file}"
                      f" ({raw_file_hash}) != {hash_value}")
                return False, None, raw_file_hash
        with open(raw_data_file, "wb") as code:
            code.write(results.content)
    except requests.exceptions.HTTPError as err:
        return False, err, None

    return results.status_code, raw_data_file, raw_file_hash

def unpack_tgz(tgz_file, dst_dir=None):
    '''Unpack a gzipped tarfile

    dst_dir: (default ".")
        destination directory for the unpack
    '''
    if dst_dir is None:
        dst_dir = pathlib.Path(".")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    tgz = tarfile.open(tgz_file, mode='r:gz')
    tgz.extractall(path=dst_dir)
