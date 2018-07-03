import numpy as np
import os
import json
import logging
import time

from src.data.utils import hash_file as _hash_file
from src.utils import record_time_interval

logger = logging.getLogger()


def save_embedding(basefilename, embedding=None, algorithm_object=None,
                   labels=None, algorithm_name=None, dataset_name=None,
                   other_info=None, run_number=0, hash_type='sha1',
                   data_path=None):
    """
    Save off a vector space embedding of a data set in a common format.

    Parameters
    ----------
    basefilename: base for the filenames
    embedding: 2d-numpy array representing embedding of points as
        rows (a row gives the coordinates for a point)
    labels: (optional) 1d-numpy array labeling the rows (points)
    algorithm_name: (str) name of the algorithm used for the given
        vector space embedding
    algorithm_object: instance of the algorithm. For example, an
        instance of sklearn.decomposition.PCA. This object should have
        a__repr__().
    dataset_name: (str) name of the dataset that was embedded
    other_info: (str) any other information to note about how
        the embedding was done
        (eg. variables for the embedding algorithm)
    run_number: (int) attempt number via the same embedding parameters
    data_path: (path) base path for save the embedding to
    """

    if embedding is None or algorithm_object is None\
       or dataset_name is None or data_path is None:
        raise ValueError("embedding, algorithm_object"
                         "dataset_name, and data_path are all required")

    metadata = create_metadata_dict(basefilename,
                                    algorithm_object=algorithm_object,
                                    algorithm_name=algorithm_name,
                                    dataset_name=dataset_name,
                                    other_info=other_info,
                                    run_number=run_number,
                                    hash_type=hash_type,
                                    data_path=data_path)
    raw_labels_filename = metadata["Labels File"].rstrip(".npy")
    raw_embedding_filename = metadata["Embedding File"].rstrip(".npy")

    filename = basefilename + "_" + str(run_number)
    metadata_filename = os.path.join(data_path, filename + ".metadata")

    embedding_shape = embedding.shape
    assert(len(embedding_shape) == 2)

    if labels is not None:
        assert(embedding_shape[0] == labels.shape[0])
        np.save(raw_labels_filename, labels)
    else:
        logger.info("No labels were given")
    labels_hashval = _hash_file(metadata["Labels File"],
                                algorithm=hash_type).hexdigest()
    metadata["Labels Hash Value"] = labels_hashval

    np.save(raw_embedding_filename, embedding)

    hashval = _hash_file(metadata["Embedding File"],
                         algorithm=hash_type).hexdigest()

    metadata["Embedding Hash Value"] = hashval

    # save metadata
    with open(metadata_filename, "w") as outfile:
        outfile.write(json.dumps(metadata, indent=4))


def read_embedding(basefilename, run_number=0, data_path=None):
    """
    Companion function for reading in a vector space embedding to
    go with save_embedding

    Returns: (embedding, labels, metadata)

    Parameters
    ----------
    basefilename:
        base for the filenames
    run_number: (int)
        attempt number via the same embedding parameters
    data_path: (path)
        base path for save the embedding to

    Returns
    -------
    (embedding, labels, metdata)
    """
    filename = basefilename + "_" + str(run_number)
    embedding_filename = os.path.join(data_path, filename + ".embedding.npy")
    labels_filename = os.path.join(data_path, filename + ".labels.npy")
    metadata_filename = os.path.join(data_path, filename + ".metadata")

    metadata = load_metadata(metadata_filename)

    logger.info(f"Reading embedding {embedding_filename}")
    embedding = np.load(embedding_filename)
    labels = np.load(labels_filename)

    assert(metadata['Run Number'] == str(run_number))

    return embedding, labels, metadata


def load_metadata(metadata_filename):
    '''
    Load a metadata file and test that the hash values match.
    '''

    with open(metadata_filename, "r") as infile:
        metadata = json.load(infile)

    hashval = _hash_file(metadata["Embedding File"],
                         algorithm=metadata['Hash Type']).hexdigest()
    labels_hashval = _hash_file(metadata["Labels File"],
                                algorithm=metadata['Hash Type']).hexdigest()

    if metadata['Embedding Hash Value'] != hashval:
        raise ValueError(f"Hash value from metadata"
                         f" {metadata['Embedding Hash Value']} "
                         f" does not match the hash of the file "
                         f" {metadata['Embedding File']}: {hashval}")
    if metadata['Labels Hash Value'] != labels_hashval:
        raise ValueError(f"Hash value from metadata"
                         f" {metadata['Labels Hash Value']} "
                         f" does not match the hash of the file "
                         f" {metadata['Labels File']}: {labels_hashval}")

    return metadata


def create_metadata_dict(basefilename, algorithm_object=None,
                         algorithm_name=None,
                         dataset_name=None, other_info=None, run_number=0,
                         hash_type='sha1', data_path=None):
    '''
    Create metadata dict that accompanies embedding files without the
    hash values for the embedding/labels files.

    Parameters
    ----------

    Returns
    -------
    metadata
    '''
    if algorithm_object is None or dataset_name is None\
       or data_path is None:
        raise ValueError("algorithm_object dataset_name, "
                         "and data_path are all required")

    if algorithm_name is None:
        algorithm_name = str(algorithm_object.__repr__()).split('(')[0]

    filename = basefilename + "_" + str(run_number)
    embedding_filename = os.path.join(data_path, filename + ".embedding")
    labels_filename = os.path.join(data_path, filename + ".labels")

    metadata = {"Algorithm": algorithm_name, "Dataset": dataset_name,
                "Run Number": str(run_number),
                "Parameters": algorithm_object.__repr__(),
                "Other Information": other_info,
                "Embedding File": embedding_filename + ".npy",
                "Labels File": labels_filename + ".npy",
                "Hash Type": hash_type}

    return metadata


def create_embedding(basefilename, algorithm_object=None,
                     labels=None, algorithm_name=None, data=None,
                     other_info=None, run_number=0, hash_type='sha1',
                     dataset_name=None, data_path=None):
    '''
    Creates an embedding and returns it, or reads it from file if it
    already exists under the data_path.

    Returns
    -------
    (embeddings, labels, metadata)
    '''
    filename = basefilename + "_" + str(run_number)
    metadata_filename = os.path.join(data_path, filename + ".metadata")

    if os.path.exists(metadata_filename):
        logger.info(f"Existing metatdata file {metadata_filename} found.")
        try:
            metadata = load_metadata(metadata_filename)
        except:
            metadata = None
    else:
        metadata = None
        logger.info("No existing metadata file found. Creating from scratch.")

    if metadata is not None:
        new_metadata = create_metadata_dict(basefilename,
                                            algorithm_object=algorithm_object,
                                            algorithm_name=algorithm_name,
                                            dataset_name=dataset_name,
                                            other_info=other_info,
                                            run_number=run_number,
                                            hash_type=hash_type,
                                            data_path=data_path)
        match = True
        for key in new_metadata.keys():
            if new_metadata[key] != metadata[key]:
                match = False
                raise ValueError(f"Desired {key}: {new_metadata[key]}"
                                 f"does not match {metadata[key]}. "
                                 "Try again with matching parameters "
                                 "or change the run number.")
        if match is True:
            logger.info(f"Desired embedding already exists. "
                        "Loading from file...")
            embedding, labels, metadata = read_embedding(basefilename,
                                                         run_number=run_number,
                                                         data_path=data_path)
            return embedding, labels, metadata


    # Create a new embedding. Assuming that algorithm_object has
    # a .fit_transform() method
    if data is None:
        raise ValueError("Data is required to create an embedding")
    ts = time.time()
    embedding = algorithm_object.fit_transform(data)
    record_time_interval(algorithm_name, ts)
    save_embedding(basefilename,
                   embedding=embedding,
                   labels=labels,
                   algorithm_object=algorithm_object,
                   algorithm_name=algorithm_name,
                   dataset_name=dataset_name,
                   other_info=other_info,
                   run_number=run_number,
                   hash_type=hash_type,
                   data_path=data_path)
    embedding, labels, metadata = read_embedding(basefilename,
                                                 run_number=run_number,
                                                 data_path=data_path)
    return embedding, labels, metadata
