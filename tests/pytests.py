

import json

import sys

from isort import file 
sys.path.append('../api')
from sensei.model import ModelCreationRequest
from sensei.api import create_model, get_model


EXPECTED_FOLDER     = './expected'
INPUT_FOLDER        = './inputs'
MODEL_OUTPUT_FOLDER = '../models'

# -- Tests

def test_pytest():
    assert True

def test_create_model():
    """
        Description
        -----------
        Test api.create_model()

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
        Test api.get_model()

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

# -- Utilities

def load_model(model_filename) -> ModelCreationRequest:
    """
        Load a model from .json file and return a ModelCreationRequest object.
    """

    model = json.load(open(model_filename))
    return ModelCreationRequest(id=model['id'], nodes=model['nodes'], edges=model['edges'])

