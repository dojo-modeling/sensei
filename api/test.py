import json
import time
import requests

# DySE SERVICE URL
SERVICE = "http://localhost:8000"

INPUT_FOLDER = '../tests/pam/609aed2f'

CAG_ID  = "bkj_001"

model       = json.load(open(f"{INPUT_FOLDER}/model.json"))
model['id'] = CAG_ID

# --

print('-' * 50)
print("create_model")

create_model_res = requests.post(f"{SERVICE}/create-model", data=json.dumps(model))
print(create_model_res.text)

# --
# !! Not working in API:

# print('-' * 50)
# print('training_progress')

# for _ in range(4):
#   progress_res = requests.get(f"{SERVICE}/models/{CAG_ID}/training-progress")
#   print(progress_res.text)


# --

print('-' * 50)
print('get_model')

get_model_res = requests.get(f"{SERVICE}/models/{CAG_ID}")
print(get_model_res.text)
time.sleep(1)

# --

def fix_proj(proj):
  """ helper function for cleaning improperly formatted projection parameters """
  
  if 'experimentParams' in proj:
    proj['experimentParam'] = proj.pop('experimentParams')
  
  proj['experimentParam']['numTimesteps'] = proj['experimentParam'].pop('numTimeSteps')
  return proj

print('-' * 50)
print('invoke_model_experiment')

proj = json.load(open(f"{INPUT_FOLDER}/projection.json"))
proj = fix_proj(proj)

experiments_res = requests.post(f"{SERVICE}/models/{CAG_ID}/experiments", data=json.dumps(proj))
print(experiments_res.text)

# --

print('-' * 50)
print('get_model_experiment')

EXPERIMENT_ID = experiments_res.json()['experimentId']

exp_result_res = requests.get(f"{SERVICE}/models/{CAG_ID}/experiments/{EXPERIMENT_ID}")
print(exp_result_res.text)

# --

