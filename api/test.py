import requests
import json
import time

# DySE SERVICE URL
SERVICE = "http://localhost:8000"

CAG_ID  = "bkj_001"

model       = json.load(open("examples/example-model.json"))
model['id'] = CAG_ID

# --

print('-' * 50)
print("create_model")

create_model_res = requests.post(f"{SERVICE}/create-model", data=json.dumps(model))
print(create_model_res.text)
time.sleep(1)

# --

print('-' * 50)
print('get_model')

get_model_res = requests.get(f"{SERVICE}/{CAG_ID}")
print(create_model_res.text)
time.sleep(1)

# --

print('-' * 50)
print('invoke_model_experiment')

proj = json.load(open("examples/example-projection.json"))
experiments_res = requests.post(f"{SERVICE}/models/{CAG_ID}/experiments", data=json.dumps(proj))
print(experiments_res.text)

# --

print('-' * 50)
print('get_model_experiment')

EXPERIMENT_ID = experiments_res.json()['experimentId']

exp_result_res = requests.get(f"{SERVICE}/models/{CAG_ID}/experiments/{EXPERIMENT_ID}")
print(exp_result_res.text)