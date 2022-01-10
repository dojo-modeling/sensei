from fastapi import FastAPI
from starlette.responses import Response
from tdmodelapi.model import EditEdgesRequest, EditEdgesResponse, ModelCreationRequest, ModelCreationResponse

tags_metadata = [

  {
    "name": "create_model",
    "description": "Create a new model by providng the INDRA statements and node/edge parameters",
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
  title="TD Model API",
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