# -*- coding: utf-8 -*-
import click
import json
import joblib
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.model_selection import GridSearchCV

#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from umap import UMAP

import numpy as np
import pandas as pd

from ..logging import logger
from ..paths import models_path, trained_models_path, processed_data_path
from ..data import datasets
from ..utils import normalize_numpy_dict, save_json
from .. import quality_measures as qm

DR_META_ESTIMATORS = {
    'grid_search': GridSearchCV
}

def available_meta_estimators():
    """Valid Meta-estimators for dimension reduction applications
    This function simply returns the list of known dimension reduction
    algorithms.

    It exists to allow for a description of the mapping for
    each of the valid strings.

    ============     ====================================
    Meta-est         Function
    ============     ====================================
    grid_search      sklearn.model_selection.GridSearchCV
    ============     ====================================
    """
    return DR_META_ESTIMATORS


DR_ALGORITHMS = {
    "autoencoder": None,
    "HLLE": LocallyLinearEmbedding(method='hessian'),
    "Isomap": Isomap(),
    "KernelPCA": KernelPCA(),
    "LaplacianEigenmaps": SpectralEmbedding(),
    "LLE": LocallyLinearEmbedding(),
    "LTSA": LocallyLinearEmbedding(method='ltsa'),
    "MDS": MDS(),
    "PCA": PCA(),
    "TSNE": TSNE(),
    "UMAP": UMAP(),
}

def available_algorithms():
    """Valid Algorithms for dimension reduction applications

    This function simply returns the list of known dimension reduction
    algorithms.

    It exists to allow for a description of the mapping for
    each of the valid strings.

    The valid quality metrics, and the function they map to, are:

    ============     ====================================
    Algorithm        Function
    ============     ====================================
    autoencoder
    isomap
    MDS
    PCA
    t-SNE
    UMAP
    ============     ====================================
    """
    return DR_ALGORITHMS


@click.command()
@click.argument('model_list')
def main(model_list, output_file='experiments.json'):
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

    trained models are written to `trained_models_path`. For every model, we write:

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


    """
    logger.info(f'Building models from {model_list}')

    os.makedirs(trained_models_path, exist_ok=True)

    with open(models_path / model_list) as f:
        training_dicts = json.load(f)

    quality_measures = qm.available_quality_measures()
    dataset_list = datasets.available_datasets()

    metadata_dict = {}
    for td in training_dicts:
        ds_name = td.get('dataset', None)
        assert ds_name in dataset_list, f'Unknown Dataset: {ds_name}'

        alg_name = td.get('algorithm', None)
        assert alg_name in DR_ALGORITHMS, f'Unknown Algorithm: {alg_name}'

        score_name = td.get('score', None)
        assert score_name in quality_measures, f'Unknown Score: {score_name}'

        meta_name = td.get('meta', None)
        if meta_name is not None:
            assert meta_name in DR_META_ESTIMATORS, f'Unknown meta-estimator: {meta_name}'

        run_number = td.get('run_number', 0)
        id_base = f"{td['algorithm']}_{td['dataset']}_{td['score']}_{run_number}"
        if id_base in metadata_dict:
            raise Exception("{id_base} already exists. Give a unique `run_number` to avoid collisions.")
        else:
            td['run_number'] = run_number
            metadata_dict[id_base] = td

    for model_key, td in metadata_dict.items():
        logger.debug(f'Creating model for {model_key}')
        ds_name = td['dataset']
        ds_opts = td.get('dataset_params', {})
        ds = datasets.load_dataset(ds_name, **ds_opts)

        alg_name = td['algorithm']
        alg_opts = td.get('algorithm_params', {})
        alg = DR_ALGORITHMS[alg_name]
        alg_default_opts = alg.get_params()

        score_name = td.get('score', None)
        score_params = td.get('score_params', {})
        assert score_name in quality_measures, f'Unknown Score: {score_name}'
        score = qm.make_hi_lo_scorer(quality_measures[score_name], **score_params)

        meta_name = td.get('meta', None)
        meta_opts = td.get('meta_params', {})
        if meta_name is not None:
            meta_alg = DR_META_ESTIMATORS[meta_name]

        if meta_name == 'grid_search':
            logger.debug(f'Grid-Searching {model_key}')
            grid_search = meta_alg(alg, alg_opts, scoring=score, **meta_opts)
            grid_search.fit(ds.data)#, y=ds.target)

            #save off the results from the grid search
            best_est = grid_search.best_estimator_ # save this off  as k.model
            joblib.dump(best_est, trained_models_path / f"{model_key}.model")

            meta_parameters_out = best_est.get_params()
            td['algorithm_params'] = meta_parameters_out
            save_json(trained_models_path / f"{model_key}.metadata", td)

            cv_results = pd.DataFrame(grid_search.cv_results_).T # save this off as k.csv
            cv_results.index.name = 'grid_search_results'
            cv_results.to_csv(trained_models_path / f"{model_key}-gridsearch.csv")

        elif meta_name is None:
            logger.debug(f'Fitting {model_key}')
            # Apply parameters straight to the estimator
            alg.set_params(**alg_opts)
            alg.fit(ds.data)
            joblib.dump(alg, trained_models_path / f"{model_key}.model")
            save_json(trained_models_path / f"{model_key}.metadata", td)

    save_json(models_path / output_file, metadata_dict)


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
