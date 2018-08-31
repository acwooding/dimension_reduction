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
from ..paths import model_path, trained_model_path
from ..data import datasets
from ..utils import save_json
from .. import quality_measures as qm
from .train import save_model
from .dr import available_algorithms
from .meta import available_meta_estimators

@click.command()
@click.argument('model_list')
@click.option('--output_file', '-o', nargs=1, type=str)
@click.option('--hash-type', '-H', type=click.Choice(['md5', 'sha1']), default='sha1')
def main(model_list, output_file='experiments.json', hash_type='sha1'):
    """Trains models speficied in the supplied `model_list` file

    output is a dictionary of trained model metadata, keyed by
    model_key := {algorithm}_{dataset}_{score}_{run_number}, where:

    dataset:
        name of dataset to use
    algorithm:
        name of algorithm (estimator) to run on the dataset
    score:
        name of scoring function (to evaluate model quality)
    run_number:
        Arbitrary integer. Combination of these 4 things must be unique.

    trained models are written to `trained_model_path`. For every model, we write:

    {model_key}.model:
        the trained model
    {model_key}.metadata:
        Metadata for this model
    {model_key}-gridsearch.csv: (optional)
        if grid-searching, this will contain the grid search results.

    Parameters
    ----------
    model_list:
        json file specifying list of meta-estimators, algorithms, and score functions to
        combine into a model
    output_file:
        name of json file to write metadata to
    hash_name:
        type of hash to use for caching


    """
    logger.info(f'Building models from {model_list}')

    os.makedirs(trained_model_path, exist_ok=True)

    with open(model_path / model_list) as f:
        training_dicts = json.load(f)

    quality_measures = qm.available_scorers()
    dataset_list = datasets.available_datasets()

    dr_algorithm_list = available_algorithms()
    meta_est_list = available_meta_estimators()

    metadata_dict = {}
    saved_meta = {}
    for td in training_dicts:
        ds_name = td.get('dataset', None)
        assert ds_name in dataset_list, f'Unknown Dataset: {ds_name}'

        alg_name = td.get('algorithm', None)
        assert alg_name in dr_algorithm_list, f'Unknown Algorithm: {alg_name}'

        score_name = td.get('score', None)
        assert score_name in quality_measures, f'Unknown Score: {score_name}'

        meta_name = td.get('meta', None)
        if meta_name is not None:
            assert meta_name in meta_est_list, f'Unknown meta-estimator: {meta_name}'

        run_number = td.get('run_number', 0)
        model_key = f"{td['algorithm']}_{td['dataset']}_{td['score']}_{run_number}"
        if model_key in metadata_dict:
            raise Exception("{id_base} already exists. Give a unique `run_number` to avoid collisions.")
        else:
            td['run_number'] = run_number
            metadata_dict[model_key] = td

    for model_key, td in metadata_dict.items():
        logger.debug(f'Creating model for {model_key}')
        ds_name = td['dataset']
        ds_opts = td.get('dataset_params', {})
        ds = datasets.load_dataset(ds_name, **ds_opts)
        td['data_hash'] = joblib.hash(ds.data, hash_name=hash_type)
        td['target_hash'] = joblib.hash(ds.target, hash_name=hash_type)

        alg_name = td['algorithm']
        alg_opts = td.get('algorithm_params', {})
        alg = dr_algorithm_list[alg_name]

        score_name = td.get('score', None)
        score_params = td.get('score_params', {})
        assert score_name in quality_measures, f'Unknown Score: {score_name}'
        score = partial(quality_measures[score_name], **score_params)

        meta_name = td.get('meta', None)
        meta_opts = td.get('meta_params', {})
        if meta_name is not None:
            meta_alg = meta_est_list[meta_name]

        if meta_name == 'grid_search':
            logger.debug(f'Grid-Searching {model_key}')
            grid_search = meta_alg(alg, alg_opts, scoring=score, **meta_opts)
            grid_search.fit(ds.data, y=ds.target)

            #save off the results from the grid search
            best_est = grid_search.best_estimator_ # save this off  as k.model
            td['algorithm_params'] = best_est.get_params()

            saved_meta[model_key] = save_model(model_name=model_key, model=best_est, metadata=td)

            cv_results = pd.DataFrame(grid_search.cv_results_).T # save this off as k.csv
            cv_results.index.name = 'grid_search_results'
            cv_results.to_csv(trained_model_path / f"{model_key}-gridsearch.csv")

        elif meta_name is None:
            logger.debug(f'Fitting {model_key}')
            # Apply parameters straight to the estimator
            alg.set_params(**alg_opts)
            alg.fit(ds.data)
            saved_meta[model_key] = save_model(model_name=model_key, model=alg, metadata=td)

    save_json(model_path / output_file, saved_meta)


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
