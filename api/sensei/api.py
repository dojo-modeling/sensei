from fastapi import FastAPI

import json
import os
from starlette.responses import Response
from sensei.model import (EditEdgesRequest, EditEdgesResponse, EdgeResponse,
  ExperimentType, ModelCreationRequest, ModelCreationResponse, Node, NodeParameter, ProjectionParameters)

# Import the logger. When running uvicorn, will need to use logger.error() to print messages.
import logging
logger = logging.getLogger(__name__)

# Base directory for saving models.
models_path = '../models'

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
    "description": "Update node parameterization for a set of nodes",
    "returns": "Ok"
  },
  {
    "name": "edit_edges",
    "description": "Edit edge weight values",
    "returns": "EditEdgesResponse"
  },
]

from collections import namedtuple

def customModelDecoder(modelDict):
    return namedtuple('X', modelDict.keys())(*modelDict.values())

def create_and_open(filename, mode):
  """
    Description
    -----------
    Creates directory prior to open attempt.

  """

  dirname = os.path.dirname(filename)
  os.makedirs(dirname, exist_ok=True)
  return open(filename, mode)

def get_experiment_filename(model_id: str, experiment_id: str):
  """
    Description
    -----------
    Standardizes experiment location based on model and experiment_ids.

  """
  

  return f'{models_path}/{model_id}//experiments/{experiment_id}/{experiment_id}.json'

def get_model_filename(model_id: str):
  """
    Description
    -----------
    Standardizes model location based on model_id.

  """

  return f'{models_path}/{model_id}/{model_id}.json'

def model_to_model_response(model) -> ModelCreationResponse:
  """
    Description
    -----------
    Generate a ModelCreationResponse object for a model passed either as a
    dict or ModelCreationRequest.
  """

  model_creation_response = ModelCreationResponse(status='success', nodes = [], edges = [])

  # Branching based on whether dict or ModelCreationRequest is passed.
  if isinstance(model, ModelCreationRequest):

    #TODO calculate scalingFactor, scalingBias
    for node in model.nodes:
      model_creation_response.nodes.append(Node(concept=node.concept, scalingFactor=0.01, scalingBias=0.5))

    #TODO calculate weights
    for edge in model.edges:
      model_creation_response.edges.append(EdgeResponse(source=edge.source, target=edge.target, weights=[]))

  else:
    #TODO calculate scalingFactor, scalingBias
    for node in model['nodes']:
      model_creation_response.nodes.append(Node(concept=node['concept'], scalingFactor=0.01, scalingBias=0.5))

    #TODO calculate weights
    for edge in model['edges']:
      model_creation_response.edges.append(EdgeResponse(source=edge['source'], target=edge['target'], weights=[]))

  return model_creation_response

# App object. Launch via command line $ uvicorn sensei.api:app
app = FastAPI(
  title="Sensei",
  description=
  """
    ## FastAPI service port of the Model Engine API for CauseMos.

    ## Endpoints
    * **Create models** (POST _/models_)
    * **Get models** (GET _/models/{model_id}_) (__not implemented__)
    * **Get model training progress** (GET _/models/{model_id}/progress_) (__not implemented__)
    * **Invoke a model experiment** (POST _/models/{model_id}/experiments_) (__not implemented__)
    * **Get a model experiment** (GET _/models/{model_id}/experiments/{experiment_id}_) (__not implemented__)
    * **Edit model nodes** (POST _/models/{model_id}/indicators_)
    * **Edit model edges** (POST _/models/{model_id}/edges_)
  """,
  version="0.0.1",
  openapi_tags = tags_metadata
  )


# Endpoints start here.

@app.post("/create-model", tags=["create_model"], response_model=ModelCreationResponse)
def create_model(payload: ModelCreationRequest) -> ModelCreationResponse:
  """
    Description
    ----------- 
        Create a new model by providng the INDRA statements and node/edge parameters
          Create a "quantified model" within the engine, given:  
            - A UUID for future reference,
            - A set of INDRA statements (subject-object relationships between concepts; could be represented by just a weighted directed adjacency matrix),
            - A set of indicator time-series data, each mapped to a concept.

          Y-Scaling:
            - DySE is a discrete model; indicator data must be converted to positive integers within a range of discrete levels. 
            - The number of levels is defined by numLevels in nodes and should be an integer of the form: 
              - numLevels = 6 * n + 1 = 7, 13, 19, ... 
            - For each concept, calculate:
              - scalingFactor = (maxValue - minValue) / ((numLevels - 1.0) / 3.0)
              - scalingBias = 0.5 * (maxValue + minValue)
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
    model_filename = get_model_filename(payload.id)
    with create_and_open(model_filename, 'w') as filehandle:
      json.dump(payload, filehandle, indent=4, default=lambda obj: obj.__dict__)

    response = model_to_model_response(payload)

    return response
  except Exception as e:
    logger.error(e)
    return ModelCreationResponse(status_code=500)


@app.get("/models/{model_id}", tags=["get_model"], response_model=ModelCreationResponse)
def get_model(model_id: str) -> ModelCreationResponse:
  """
  Description
  -----------
    Return model status.
  """

  try:
    # Get the model filepath based on the model_id.
    model_filename = get_model_filename(model_id)

    # Load the model.
    try:
      with open(model_filename, 'r') as filehandle:
        model=json.load(filehandle)

      return model_to_model_response(model)

    except FileNotFoundError as e:
      logger.error(f'ERROR:     Could not find file for model_id {model_id}')
      return Response(status_code=404, content="Model not found.")

  except Exception as e:
    logger.error(e)
    return Response(status_code = 500)


@app.get("/models/{model_id}/training-progress", tags=["get_model_training_progress"])
def get_model_training_progress(model_id: str) -> float:
    return Response(status_code=200, content={'progress': 0.97})


@app.post("/models/{model_id}/experiments", tags=["invoke_experiment"]) #, response_model=InvokeExperimentResponse)
def invoke_model_experiment(model_id: str, payload: ProjectionParameters): # -> InvokeExperimentResponse:
  """
    Description
    -----------
        The payload varies deends on the experiment type, which can be one of [PROJECTION, SENSITIVITY]. This should return immediately with a designated experimentId while the actual experiment runs in the background

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

  # ProjectionParameters and SensitivityAnalysisParameters should probably be 
  # consolidated into a single model with experimentType determining flow.
  try:
    if payload.experimentType == ExperimentType.PROJECTION:
      logger.error(f'INFO:     Run experiment with projection parameters {payload.experimentParams}')
      
      # TODO: do something way cool async
      
      return {'experimentId': 'uuid'}
    else:
      # Uvicorn is probably going to blow a 404 based on the model expermentType before it gets here.
      return Response(status_code = 404, content=f"Experiment type {payload.experimentType} not supported")
  except Exception as e:
    logger.error(e)
    return Response(status_code = 500)


@app.get("/models/{model_id}/experiments/{experiment_id}", tags=["get_experiment"])
def get_model_experiment(model_id: str, experiment_id: str):
  """
    Description
    -----------

    Returns ProjectionResponse or SensitivityAnalysisResponse
  """

  # Get the experiment results filepath based on the model and experiment ids.
  experiment_filename = get_experiment_filename(model_id, experiment_id)

  logger.error(experiment_filename)

  #experiment_filename = "/home/user/source/repos/sensei/models/dyse-graph-like1/experiments/dyse-graph-like1__59fd2df373/dyse-graph-like1__59fd2df373.json"
  # Load the experiment.
  try:
    try:
      with open(experiment_filename, 'r') as filehandle:
        experiment = json.load(filehandle)

      return experiment

    except FileNotFoundError as e:
      logger.error(f'ERROR:     Could not find experiment results {experiment_id} for model_id {model_id} ')
      return Response(status_code=404, content=f"Experiment {experiment_id} for model_id {model_id} not found.")
  
  except Exception as e:
    logger.error(e)
    return Response(status_code = 500)


@app.post("/models/{model_id}/edit-nodes", tags=["edit_nodes"])
def edit_nodes(model_id: str, payload: NodeParameter):
  """
  Description
  -----------
    Replace a model's nodes JSON with the posted JSON.

  """

  try:
    # Get the model filepath based on the model_id.
    model_filename = get_model_filename(model_id)

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
      
    return Response(status_code=200)

  except Exception as e:
    logger.error(e)
    return Response(status_code=500)

  
@app.post("/models/{model_id}/edit-edges", tags=["edit_edges"], response_model=EditEdgesResponse)
def edit_edges(model_id: str, payload: EditEdgesRequest):
  """
  Description
  -----------
    Replace a model's edges JSON with the posted JSON.

  """

  try:
    # Get the model filepath based on the model_id.
    model_filename = get_model_filename(model_id)

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

    return EditEdgesResponse(status_code=200)
    
  except Exception as e:
    logger.error(e)
    return Response(status_code=500)

  
