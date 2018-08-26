# -*- coding: utf-8 -*-
import click
from src.logging import logger
import os
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from ..paths import models_path, trained_models_path

@click.command()
@click.argument('model_list')
def main(action):
    """Trains models speficied in the model list
    trained models are written to `trained_models_path`

    """
    logger.info(f'Training on {model_list}')

    os.makedirs(trained_models_path, exist_ok=True)

    with open(models_path / model_list) as f:
        training_dicts = json.read(f)

    for td in training_dicts:
        pass


if __name__ == '__main__':

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
