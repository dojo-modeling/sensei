
"""

Usage:

/sensei/tests$ pytest pytests.py

"""

import json
import sys

sys.path.append('../api')
from sensei.model import ModelCreationRequest, ProjectionParameters
from sensei.api import create_model, get_model, get_model_experiment, invoke_model_experiment


# -- Directories

EXPECTED_FOLDER     = './expected'
INPUT_FOLDER        = './inputs'
MODEL_OUTPUT_FOLDER = '../models'

EXPERIMENT_ID_FILE  = f'{EXPECTED_FOLDER}/experimentId.txt'

# -- Tests

def test_pytest():
    assert True

def test_create_model():
    """
        Description
        -----------
        (1) Test api.create_model().

        (3) Compare the model output to expected model output files in /expected.

    """
    
    # Create the ModelCreationRequest from the json file.
    model = load_model(f'{INPUT_FOLDER}/example-model.json')

    # Call api.create_model() which returns a ModelCreationResponse object.
    response = create_model(payload = model)

    # Convert the ModelCreationResponse object to a json string.
    test_response = json.dumps(response, default=lambda obj: obj.__dict__)

    # Load the expected response.
    expected_response = json.load(open(f'{EXPECTED_FOLDER}/test_create_model.json'))

    # Assert the responses are equal.
    assert test_response == expected_response

    # Iterate the model output .csv files and assert these are as expected.
    for output_file in ['df_cag.csv', 'df_cag_opt.csv', 'df_ts.csv']:

        with open(f'{MODEL_OUTPUT_FOLDER}/{model.id}/{output_file}') as fh:
            test_data = fh.readlines()

        with open(f'{EXPECTED_FOLDER}/test_create_model.{output_file}') as fh:
            expected_data = fh.readlines()

        assert test_data == expected_data

def test_get_model():
    """
        Description
        -----------
        (1) Test api.get_model().

        (2) Compare to the expected model.json in /exptected.

    """

    # Get the model id from the example-model.
    model_id = json.load(open(f'{INPUT_FOLDER}/example-model.json'))['id']

    # Call api.get_model() which returns a ModelCreationResponse object.
    response = get_model(model_id)

    # Convert the ModelCreationResponse object to a dict.
    test_response = json.loads(json.dumps(response, sort_keys= True, default=lambda obj: obj.__dict__))

    # Load the expected response.
    expected_response = json.load(open(f'{EXPECTED_FOLDER}/test_get_model.json'))

    # Assert the responses are equal.
    assert test_response == expected_response

def test_invoke_model_experiment():
    """
        Description
        -----------
        (1) Test api.invoke_model_experiment()

    """

    # Get the model id from the example-model.
    model_id = json.load(open(f'{INPUT_FOLDER}/example-model.json'))['id']

    # Get the projection.
    projection = load_experiment(f"{INPUT_FOLDER}/example-projection.json")

    # Invoke experiment.
    experiments_res = invoke_model_experiment(model_id, projection)

    # Save experiment_id for test_get_model_experiment().
    with open(EXPERIMENT_ID_FILE, 'w') as fh:
        fh.write(experiments_res['experimentId'])
      
def test_get_model_experiment():
    """
        Description
        -----------
        (1) Test api.invoke_model_experiment()

    """

    # Get the model id from the example-model.
    model_id = json.load(open(f'{INPUT_FOLDER}/example-model.json'))['id']

    # Load the experimentId from the previous test of invoke_experiment.
    with open(EXPERIMENT_ID_FILE, 'r') as fh:
        experimentId = fh.readline().strip()

    get_model_experiment(model_id, experimentId)


# -- Utilities

def load_experiment(experiment_filename) -> ProjectionParameters:
    """
        Load an experiment from .json file and return a ProjectionParameters object.
    """

    exp = json.load(open(experiment_filename))
    return ProjectionParameters(experimentType=exp['experimentType'], experimentParams=exp['experimentParams'])

def load_model(model_filename) -> ModelCreationRequest:
    """
        Load a model from .json file and return a ModelCreationRequest object.
    """

    model = json.load(open(model_filename))
    return ModelCreationRequest(id=model['id'], nodes=model['nodes'], edges=model['edges'])
