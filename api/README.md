# sensei api

v0.2.3

#### FastAPI service port of the Model Engine API for CauseMos.

## Endpoints
* [**Create models**](#/create_model) (POST _/models_)
* [**Get models**](#/models/{model_id}) (GET _/models/{model_id}_) (__not implemented__)
* **Get model training progress** (GET _/models/{model_id}/progress_) (__not implemented__)
* [**Invoke a model experiment**](#/models/{model_id}/experiments) (POST _/models/{model_id}/experiments_)
* [**Get a model experiment**](/models/{model_id}/experiments/{experiment_id}) (GET _/models/{model_id}/experiments/{experiment_id}_)
* **Edit model nodes** (POST _/models/{model_id}/indicators_)
* [**Edit model edges**](/models/{model_id}/edit-edges) (POST _/models/{model_id}/edges_)
"""


## Installation

### Local Install
```
pip install -r requirements.txt
conda install uvicorn
```

### Docker Install

```
$ docker-compose up --build
```

## Launching the API

From /sensei/api repo directory:
```
uvicorn sensei.api:app
```

Test/view the API locally at:
```
http://127.0.0.1:8000/
```

Use port 8888 if running in Docker.

## Endpoints

## */create_model*

Create a new model.

### Parameters
- None

### Request body

JSON representation of [api.model.ModelCreationRequest](sensei/model.py#L265)


### Response body

JSON representation of [api.model.ModelCreationResponse](sensei/model.py#L101) e.g.:
```
{
  "status": "success",
  "nodes": [
    {
      "concept": "wm/process/conflict/attackdiediedie",
      "scalingFactor": 0.01,
      "scalingBias": 0.5
    },
    {
      "concept": "wm/process/population/death",
      "scalingFactor": 0.01,
      "scalingBias": 0.5
    }
  ],
  "edges": [
    {
      "source": "wm/process/conflict/attackdiediedie",
      "target": "wm/process/population/death",
      "weights": []
    }
  ]
}
```

## */models/{model_id}*

Return model status.

### Parameters
- `--model_id` : model id e.g. *dyse-graph-like1*

### Request body
- None

### Response body

JSON representation of [api.model.ModelCreationResponse](sensei/model.py#L101) (same response body as [/create_models](#/create_model)).

## */models/{model_id}/experiments*

Invoke a model experiment.

### Parameters
- `--model_id` : model id e.g. *dyse-graph-like1*

### Request body

JSON representation of [api.model.ProjectionParameters](sensei/model.py#L132) e.g.:
```
{
  "experimentType": "PROJECTION",
  "experimentParam": {
    "numTimesteps": 12,
    "startTime": 1609459200000,
    "endTime": 1638316800000,
    "constraints": [
    ]
  }
}
```

### Response body

The experimentId uuid is returned: `{ experimentId: '59fd2df373' }`


## */models/{model_id}/experiments/{experiment_id}*

### Parameters
- `--model_id` : model id e.g. *dyse-graph-like1*
- `--experiment_id` : experiment id e.g. *59fd2df373*

## Request body
- None

### Response body

JSON representation of [api.model.ProjectionResponse](sensei/model.py#L201) e.g.:
```
{
  "modelId": "dyse-graph-like1",
  "experimentType": "PROJECTION",
  "experimentId": "59fd2df373",
  "status": "Completed",
  "progressPercentage": 1,
  "results": {
    "data": [
      {
        "concept": "wm/process/conflict/attack",
        "values": [
          {
            "timestamp": 1609459200000,
            "values": [
              8157.452387096775,
              8157.452387096775,
...
```

## */models/{model_id}/edit-nodes*

Replace a individual model node with the posted NodeParameter JSON.

### Parameters
- `--model_id` : model id e.g. *dyse-graph-like1*

## Request body

JSON representation of [api.model.NodeParameter](sensei/model.py#L243) e.g.:
```
{
  "edges": [
    {
      "source": "wm/process/conflict/attack",
      "target": "wm/process/population/death",
      "polarity": -11.0,
      "weights": [99,98]
    }
  ]
}
```

### Response body

- `{ "status": "success" }`

## */models/{model_id}/edit-edges*

Modify model edges polarity and weights (but not statements) matched by source/target.

### Parameters
- `--model_id` : model id e.g. *dyse-graph-like1*

## Request body

JSON representation of [api.model.EditEdgesRequest](sensei/model.py#L225) e.g.:
```
{
  "edges": [
    {
      "source": "wm/process/conflict/attack",
      "target": "wm/process/population/death",
      "polarity": -11.0,
      "weights": [99,98]
    }
  ]
}
```

### Response body

- `{ "status": "success" }`
