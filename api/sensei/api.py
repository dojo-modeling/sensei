from pydantic.utils import Obj
from fastapi import FastAPI

import json
import os
from starlette.responses import Response
from sensei.model import (BaseModel, EditEdgesRequest, EditEdgesResponse, EditNodesRequest,
  ModelCreationRequest, ModelCreationResponse, ProjectionParameters, Status)

# Import the logger. When running uvicorn, will need to use logger.error() to print messages.
import logging
logger = logging.getLogger(__name__)

# Base directory for saving models.
models_path = 'models'

# http: 8000/docs descriptions.
tags_metadata = [

  {
    "name": "create_model",
    "description": "Create a new model",
    "returns": "ModelCreationResponse"
  },
  {
    "name": "get_model",
    "description": "Returns model status",
    "returns": "ModelCreationResponse"
  },
  {
    "name": "get_model_training_progress",
    "description": "Returns the model training progress as a percentage",
    "returns": "number"
  },
  {
    "name": "invoke_experiment",
    "description": "Invoke an experiment.",
    "returns": "Experiment Id"
  },
  {
    "name": "get_experiment",
    "description": "Retrieves the state and result of a modeling experiment.",
    "returns": "One of ProjectionResponse or SensitivityAnalysisResponse"
  },
  {
    "name": "edit_nodes",
    "description": "Update indicator specification for a set of nodes",
    "returns": "Ok"
  },
  {
    "name": "edit_edges",
    "description": "Edit edge weight values",
    "returns": "EditEdgesResponse"
  },
]

def create_and_open(filename, mode):
  """
    Description
    -----------
    Creates directory prior to open attempt.

  """

  dirname = os.path.dirname(filename)
  os.makedirs(dirname, exist_ok=True)
  return open(filename, mode)

def model_id_to_filename(id: str):
  """
    Description
    -----------
    Standardizes model location based on model_id.

  """

  return f'{models_path}/{id}/{id}.json'


description = """
## FastAPI service port of the Model Engine API for CauseMos.

## Endpoints
* **Create models** (POST _/models_)
* **Get models** (GET _/models/{model_id}_) (__not implemented__)
* **Get model training progress** (GET _/models/{model_id}/progress_) (__not implemented__)
* **Invoke a model experiment** (POST _/models/{model_id}/experiments_) (__not implemented__)
* **Get a model experiment** (GET _/models/{model_id}/experiments/{experiment_id}_) (__not implemented__)
* **Edit model nodes** (POST _/models/{model_id}/indicators_)
* **Edit model edges** (POST _/models/{model_id}/edges_)
"""

# App object. Launch via command line $ uvicorn sensei.api:app
app = FastAPI(
  title="Sensei",
  description=description, #"FastAPI service port of the Model Engine API for Causemos",
  version="0.0.1",
  openapi_tags = tags_metadata
  )


@app.post("/models", tags=["create_model"], response_model=ModelCreationResponse)
def create_model(payload: ModelCreationRequest) -> ModelCreationResponse:
  """
    Description
    ----------- 
        Create a new model by providng the INDRA statements and node/edge parameters.

        Create a "quantified model" within the engine, given:  
        - A UUID for future reference,
        - A set of INDRA statements (subject-object relationships between concepts; could be represented by just a weighted directed adjacency matrix),
        - A set of indicator time-series data, each mapped to a concept.

        Aggregation:
            - func in conceptIndicators specifies the aggregation function to be applied on the indicator data to calculate the initial value of each concept. 
            - Allowed values: {"min", "max", "first", "last", "median"}, where the default value is "last".
        
        Y-Scaling:
            - DySE is a discrete model; indicator data must be converted to positive integers within a range of discrete levels. 
            - The number of levels is defined by numLevels in conceptIndicators and should be an integer of the form: 
              - numLevels = 6 * n + 1 = 7, 13, 19, ... 
            - For each concept, calculate:
              - scalingFactor = (maxValue - minValue) / ((numLevels - 1.0) / 3.0)
              - scalingBias = 0.5 * (maxValue + minValue)
              - Note: the scalingFactor was (maxValue - minValue) / ((numLevels - 1.0) / 3.0), but now that we allow the user to set the max and min values, we don't divide by 3
            - Discretize and standardize the indicator time-series data for each concept by calculating
              - indValues_ = np.floor((indValues - scalingBias) / scalingFactor + 0.5 * numLevels)
            - indValues is the array of values of each indicator time-series data sent by Causemos to be converted; indValues_ is the array of converted values to be sent to the engine.
            - For out-of-range values, 
              - indValues_ = numLevels - 1    (if indValues > 2 * maxValue - minValue)
              - indValues_ = 0                                (if indValues < 2 * minValue - maxValue)
            - scalingFactor, scalingBias, and numLevels should be stored for reverse y-scaling in the results returned by subsequent requests.
            - If no indicator data is available, Causemos will send
              - minValue = 0.0
              - maxValue = 1.0
              - values: []
              - The Causemos initial value should be 0.5 for any aggregation function "func". Apply y-scaling and reverse y-scaling as usual to get the DySE discrete values.
            - This scheme is defined only for numLevels >= 7; if given a smaller integer, default to 7. 
            - For Delphi, this could be implemented or bypassed completely.
  
  """

  try:
    model_filename = model_id_to_filename(payload.id)
    with create_and_open(model_filename, 'w') as filehandle:
      json.dump(payload, filehandle, indent=4, default=lambda obj: obj.__dict__)

    return ModelCreationResponse(status_code=200)
  except Exception as e:
    logger.error(e)
    return ModelCreationResponse(status_code=500)


@app.get("/models/{model_id}", tags=["get_model"], response_model=ModelCreationResponse)
def get_model(model_id: str):
  # What is this supposed to do?
  pass
  """
  filename = model_id_to_filename(model_id)

  try:
    if os.path.exists(filename):
      try:
        return json.load(filename)
      except Exception as e:
        return ModelCreationResponse(status = 404)
  except Exception as e:
    return ModelCreationResponse(status = 500)
    """


@app.get("/models/{model_id}/training-progress", tags=["get_model_training_progress"])
def get_model_training_progress(model_id: str) -> float:
    return Response(status_code=200, content={'progress': 0.97})


@app.post("/models/{model_id}/experiments", tags=["invoke_experiment"])
def invoke_model_experiment(model_id: str, payload: BaseModel):
  """
    Description
    -----------
    The payload varies depends on the experiment type, which can be one of [PROJECTION, SENSITIVITY]. This should return immediately with a designated experimentId while the actual experiment runs in the background

            PROJECTION:
              - Generate a timeseries projection of each indicator based on the given model, parameters, and constraints.
              - Each timestep of the series contains an array of numbers representing the full distribution of projected values.

            GOAL_OPTIMIZATION:
              - Perform optimization over the initial values of the model to ensure that the projections achieve given fixed values or "goals".
              - As in the case of the projection constraints, the goal values need to be y-scaled before being input in the optimizer and the solution values need to be reverse y-scaled before being returned to CauseMos. 
              - DySE currently uses linear programming to perform this experiment according to https://drive.google.com/file/d/1E4wL1JE8q_seQvCXJz7pMoJ0eVhFk84s/view?usp=sharing

            SENSITIVITY_ANALYSIS:
              - Perform graph-theoretic calculations on the model graph.
              - Three types of analysis are defined: analysisType ∈ {"IMMEDIATE", "GLOBAL", "PATHWAYS"}
              - "IMMEDIATE" and "GLOBAL" requests return the "influence" from one given set of concepts in the model to another.
              - "IMMEDIATE" vs. "GLOBAL": either only the immediate neighbours or all nodes in the model graph are considered in the influence calculation.
              - "PATHWAYS" requests search, score, sort, and return the top numPath pathways connecting the source and target nodes in the model graph.
              - pathAtt ∈ {"INFLUENCE", "SENSITIVITY"} specifies how the scores are calculated and it corresponds to the "influ" and "sensi" options internal to DySE.
              - numPath is recommended to be < 5 to avoid excessively long runtimes.
              - Two modes of analysis are defined: analysisMode ∈ {"STATIC", "DYNAMIC"}
              - If source = [] and target = [], then return the influence for all concepts as source and/or target, i.e. return the row, col, or all of the source-target influence matrix.
              - DySE is capable of doing these calculations for a given scenario in "DYNAMIC" mode but this is not defined here (yet).
              - See https://arxiv.org/abs/1902.03216
              - See https://drive.google.com/file/d/1eiXiYmJIA66G7Fxt8ZbBjyepeE97YRNq/view?usp=sharing
    
    Returns
    -------
      experimentId: type: string format: uuid
  """

  logger.error(f'payload is {type(payload)}')
  return Response(status_code = 200)

  """try:

     if isinstance(payload, ProjectionParameters):
      logger.error(f'INFO:      Run experiment with projection parameters {payload.experimentParams}')
      # TODO: do something
      return Response(status_code = 200, experimentId="uuid here")
    else:
      return Response(status_code = 404, content="Exeriment type not supported")
  except Exception as e:
    logger.error(e)
    return Response(status_code = 500)"""



@app.get("/models/{model_id}/experiments/{experiment_id}", tags=["get_experiment"])
def get_model_experiment(model_id: str, experiment_id: str):
  pass


@app.post("/models/{model_id}/edit-indicators", tags=["edit_nodes"])
def edit_nodes(model_id: str, payload: EditNodesRequest):
  """
  Description
  -----------
    Replace a model's nodes JSON with the posted JSON.

  """

  try:
    # Get the model filepath based on the model_id.
    model_filename = model_id_to_filename(model_id)

    # Load the model.
    try:
      with open(model_filename, 'r') as filehandle:
        model= json.load(filehandle)
    except FileNotFoundError as e:
      logger.error(f'ERROR:     Could not find file for model_id {model_id}')
      return Response(status_code=404, content="Model not found.")

    # Pull the switcheroo.
    model['nodes'] = payload.nodes

    # Write the modfiied model to file.
    with create_and_open(model_filename, 'w') as filehandle:
      json.dump(model, filehandle, indent=4, default=lambda obj: obj.__dict__)

  except Exception as e:
    logger.error(e)
    return Response(status_code = 500)

  return EditEdgesResponse(status_code=200)


@app.post("/models/{model_id}/edit-edges", tags=["edit_edges"], response_model=EditEdgesResponse)
def edit_edges(model_id: str, payload: EditEdgesRequest):
  """
  Description
  -----------
    Replace a model's edges JSON with the posted JSON.

  """

  try:
    # Get the model filepath based on the model_id.
    model_filename = model_id_to_filename(model_id)

    # Load the model.
    try:
      with open(model_filename, 'r') as filehandle:
        model= json.load(filehandle)
    except FileNotFoundError as e:
      logger.error(f'ERROR:     Could not find file for model_id {model_id}')
      return Response(status_code=404, content="Model not found.")

    # Pull the switcheroo.
    model['edges'] = payload.edges

    # Write the modfiied model to file.
    with create_and_open(model_filename, 'w') as filehandle:
      json.dump(model, filehandle, indent=4, default=lambda obj: obj.__dict__)

  except Exception as e:
    logger.error(e)
    return Response(status_code=500)

  return EditEdgesResponse(status_code=200)