# sensei api

#### FastAPI service port of the Model Engine API for CauseMos.

## Endpoints
* **Create models** (POST _/models_)
* **Get models** (GET _/models/{model_id}_) (__not implemented__)
* **Get model training progress** (GET _/models/{model_id}/progress_) (__not implemented__)
* **Invoke a model experiment** (POST _/models/{model_id}/experiments_) (__not implemented__)
* **Get a model experiment** (GET _/models/{model_id}/experiments/{experiment_id}_) (__not implemented__)
* **Edit model nodes** (POST _/models/{model_id}/indicators_)
* **Edit model edges** (POST _/models/{model_id}/edges_)
"""


## Installation
```
pip install -r requirements.txt
conda install uvicorn
```

## Launching the API

From sensei repo directory:
```
uvicorn sensei.api:app
```

```
http://127.0.0.1:8000/docs#/
```
