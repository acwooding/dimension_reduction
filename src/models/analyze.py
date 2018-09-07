import inspect

from .. import quality_measures as qm
from ..paths import model_output_path
from .predict import load_prediction
from ..data import Dataset


def get_score(run_number=0, file_base=None, input_path=None, score_params=None,
              pointwise=False, *dataset_name, model_name, score_name):
    '''
    Given a dataset_name, model_name and score_name, compute the given score
    on the output of the model on the specified dataset.
    '''
    if input_path is None:
        input_path = model_output_path

    if file_base is None:
        file_base = f'{model_name}_exp_{dataset_name}_{run_number}'

    score = qm.available_quality_measures(pointwise=pointwise)['score_name']

    low_data = load_prediction(predict_name=file_base, predict_path=input_path)

    # check if our score requires high_data
    if 'high_data' in inspect.getfullargspec(score).args:
        high_data = Dataset.load_dataset(dataset_name)
        s = score(low_data=low_data, high_data=high_data, **score_params)
    else:
        s = score(low_data=low_data, **score_params)

    return s
