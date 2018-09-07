# -*- coding: utf-8 -*-
import click
import json
import joblib
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from functools import partial

import numpy as np
import pandas as pd

from ..logging import logger
from ..paths import model_path, trained_model_path, model_output_path
from ..data import datasets
from ..utils import save_json
from .. import quality_measures as qm
from . import save_model
from . import available_algorithms
from . import available_meta_estimators
from . import run_model

@click.command()
@click.argument('model_list')
@click.option('--output_file', '-o', nargs=1, type=str)
@click.option('--hash-type', '-H', type=click.Choice(['md5', 'sha1']), default='sha1')
def main(model_list, output_file='predictions.json', hash_type='sha1'):
    logger.info(f'Executing models from {model_list}')

    os.makedirs(model_output_path, exist_ok=True)

    with open(model_path / model_list) as f:
        predict_list = json.load(f)

    saved_meta = {}
    metadata_keys = ['dataset_name', 'descr', 'hash_type', 'data_hash', 'target_hash', 'experiment']
    for exp in predict_list:
        ds = run_model(**exp)
        name = ds.metadata['dataset_name']
        metadata = {}
        for key in metadata_keys:
            metadata[key] = ds.metadata[key]
            saved_meta[name] = metadata

    save_json(model_path / output_file, saved_meta)

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
