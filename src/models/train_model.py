# -*- coding: utf-8 -*-
import click
from ..logging import logger
import json
import joblib
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ..paths import models_path, trained_models_path, processed_data_path
from ..data import datasets
from ..utils import normalize_numpy_dict, save_json

from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap, LocallyLinearEmbedding, SpectralEmbedding
from sklearn.model_selection import GridSearchCV

#from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.manifold import TSNE
from umap import UMAP

import numpy as np
import pandas as pd

from .. import quality_measures as qm

meta_estimators = {
    'grid_search': GridSearchCV
}

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
def main(model_list):
    """Trains models speficied in the model list
    trained models are written to `trained_models_path`

    """
    logger.info(f'Building models from {model_list}')

    os.makedirs(trained_models_path, exist_ok=True)

    with open(models_path / model_list) as f:
        training_dicts = json.load(f)

    metadata_dict = {}
    for td in training_dicts:
        assert td['dataset'] in datasets.available_datasets()
        assert td['algorithm'] in available_algorithms()
        assert td['score'] in qm.available_quality_measures()
        run_number = td.get('run_number', 0)
        id_base = f"{td['algorithm']}_{td['dataset']}_{td['score']}_{run_number}"
        if id_base in metadata_dict:
            raise Exception("{id_base} already exists. Give a run_number to avoid collisions.")
        else:
            metadata_dict[id_base] = td

    for k, td in metadata_dict.items():
        logger.debug(f'Creating model for {k}')
        meta = td.get('meta', None)
        if meta == 'grid_search':
            ds = datasets.load_dataset(td['dataset'])
            alg = available_algorithms()[td['algorithm']]
            score = qm.make_hi_lo_scorer(qm.available_quality_measures()[td['score']], **td['score_params'])
            grid_search = meta_estimators[td['meta']](alg, td['algorithm_params'], scoring=score, **td['meta_params'])
            grid_search.fit(ds.data)#, y=ds.target)

            #save off the results from the grid search
            print(k) # metadata id
            metadata = td.copy()
            metadata['algorithm_params'] = normalize_numpy_dict(grid_search.best_params_) # save td off as k.metadata
            save_json(trained_models_path / f"{k}.metadata", metadata)
            best_est = grid_search.best_estimator_ # save this off  as k.model
            joblib.dump(best_est, trained_models_path / f"{k}.model")
            cv_results = pd.DataFrame(grid_search.cv_results_).T # save this off as k.csv
            cv_results.index.name = 'grid_search_results'
            cv_results.to_csv(trained_models_path / f"{k}-gridsearch.csv")

if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
