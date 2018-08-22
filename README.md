[![Coverage Status](https://coveralls.io/repos/github/acwooding/dimension_reduction/badge.svg?branch=coveralls)](https://coveralls.io/github/acwooding/dimension_reduction?branch=coveralls)

Python Dimension Reduction Playground
=====================================

Does your dimension reduction do what you think it does? Let's find out.


GETTING STARTED
---------------

* Create and switch to the virtual environment. This environment will
  contain everything necessary to run the project.

```
make create environment
```


* Activate the environment
```
conda activate dimension_reduction
```

* Fetch the raw data and process it into a usable form
```
make data
```

* Explore the notebooks in the `notebooks` directory

UPDATING DEPENDENCIES
---------------------

* If you find later on that you're missing something in your environment, add it to
  `environment.yml` and then run

`make requirements`

Project Organization
------------

* `LICENSE`
* `Makefile`
    * top-level makefile. Type `make` for a list of valid commands
* `README.md`
    * this file
* `data`
    * Data directory. often symlinked to a filesystem with lots of space
    * `data/raw`
        * Raw (immutable) hash-verified downloads
    * `data/interim`
        * Extracted and interim data representations
    * `data/processed`
        * The final, canonical data sets for modeling.
* `docs`
    * A default Sphinx project; see sphinx-doc.org for details
* `models`
    * Trained and serialized models, model predictions, or model summaries
* `notebooks`
    *  Jupyter notebooks. Naming convention is a number (for ordering),
    the creator's initials, and a short `-` delimited description,
    e.g. `1.0-jqp-initial-data-exploration`.
* `references`
    * Data dictionaries, manuals, and all other explanatory materials.
* `reports`
    * Generated analysis as HTML, PDF, LaTeX, etc.
    * `reports/figures`
        * Generated graphics and figures to be used in reporting
* `requirements.txt`
    * (if using pip+virtualenv) The requirements file for reproducing the
    analysis environment, e.g. generated with `pip freeze > requirements.txt`
* `environment.yml`
    * (if using conda) The YAML file for reproducing the analysis environment
* `setup.py`
    * Turns contents of `src` into a
    pip-installable python module  (`pip install -e .`) so it can be
    imported in python code
* `src`
    * Source code for use in this project.
    * `src/__init__.py`
        * Makes src a Python module
    * `src/data`
        * Scripts to fetch or generate data. In particular:
        * `src/data/make_dataset.py`
            * Run with `python -m src.data.make_dataset fetch`
            or  `python -m src.data.make_dataset process`
    * `src/features`
        * Scripts to turn raw data into features for modeling, notably `build_features.py`
    * `src/models`
        * Scripts to train models and then use trained models to make predictions.
        e.g. `predict_model.py`, `train_model.py`
    * `src/visualization`
        * Scripts to create exploratory and results oriented visualizations; e.g.
        `visualize.py`
* `tox.ini`
    * tox file with settings for running tox; see tox.testrun.org


--------

Project based on `cookiecutter-easydata`, which is an experimental fork of
`cookiecutter-data-science`
