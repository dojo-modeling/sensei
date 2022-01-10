from fastapi import FastAPI
from starlette.responses import Response
from sensei.model import EditEdgesRequest, EditEdgesResponse, ModelCreationRequest, ModelCreationResponse

tags_metadata = [

  {
    "name": "create_model",
    "description": 
    """
    Description 
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
    
    """,
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
    "description": "Invoke an experiment",
    "returns": "Experiment Id"
  },
  {
    "name": "get_experiment",
    "description": "Retrieves the state and result of a modeling experiment.",
    "returns": "One of ProjectionResponse or SensitivityAnalysisResponse"
  },
  {
    "name": "edit_indicators",
    "description": "Update indicator specification for a set of nodes",
    "returns": "Ok"
  },
  {
    "name": "edit_edges",
    "description": "Edit edge weight values",
    "returns": "EditEdgesResponse"
  },
]

app = FastAPI(
  title="Sensei",
  description="FastAPI service port of the Model Engine API for Causemos",
  version="0.0.1",
  openapi_tags = tags_metadata
  )


@app.post("/models", tags=["create_model"], response_model=ModelCreationResponse)
def create_model(payload: ModelCreationRequest) -> ModelCreationResponse:
    return ModelCreationResponse(status = '200')


@app.get("/models/{model_id}", tags=["get_model"], response_model=ModelCreationResponse)
def get_model(model_id: str):
    pass


@app.get("models/{model_id}/training-progress", tags=["get_model_training_progress", "models"])
def get_model_training_progress(model_id: str) -> float:
    return Response(status_code=200, content={'progress': 0.97})


@app.post("models/{model_id}/experiments", tags=["invoke_experiment"])
def invoke_model_experiment(model_id: str, payload):
    # Payload can be one of ProjectionParameters or SensitivityAnalysisParameters
    # Returns experimentId: type: string format: uuid
    pass


@app.get("/models/{model_id}/experiments/{experiment_id}", tags=["get_experiment"])
def get_model_experiment(model_id: str, experiment_id: str):
    pass


@app.post("/models/{model_id}/indicators", tags=["edit_indicators"])
def post_model_indicators(model_id:str, payload):
    pass


@app.post("/models/{model_id}/edit-edges", tags=["edit_edges"], response_model=EditEdgesResponse)
def edit_model_edges(model_id: str, payload: EditEdgesRequest):
    #returns EditEdgesResponse
    pass