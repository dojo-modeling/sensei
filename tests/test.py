import os
import sys
import json
import arrow
import requests
from time import sleep
from copy import deepcopy

def test_api(
  model,
  proj,
  cag_id   = None,
  # service  = 'https://sensei.dojo-modeling.com/sensei',
  service  = 'http://0.0.0.0:8000',
  username = 'wmuser',
  password = 'sunsetmoth',
):
  model = deepcopy(model)
  proj  = deepcopy(proj)
  
  auth = requests.auth.HTTPBasicAuth(username, password)
  
  if cag_id is None:
    cag_id = f'jataware_dev_{int(arrow.utcnow().timestamp())}'
  
  model['id'] = cag_id
  
  print('test_api: create model', file=sys.stderr)
  
  create_json  = requests.post(
    f"{service}/create-model", 
    auth = auth, 
    data = json.dumps(model)
  ).json()
  
  sleep(1)
  
  print('test_api: run projection', file=sys.stderr)
  proj_json = requests.post(
    f"{service}/models/{cag_id}/experiments", 
    auth = auth, 
    data = json.dumps(proj)
  ).json()
  
  experiment_id = proj_json['experimentId']
  
  print('test_api: get results', file=sys.stderr)
  while True:
    sleep(1)
    print('waiting...', file=sys.stderr)
    
    res_json = requests.get(
      f"{service}/models/{cag_id}/experiments/{experiment_id}",
      auth = auth
    ).json()
    
    if (res_json['status'] == 'Completed') and ('results' in res_json):
      break
  
  return create_json, res_json

# --

sys.path.append('../engine')

from rcode import *
from matplotlib import pyplot as plt

import numpy as np
from sensei_engine.causemos_parsers import fix_proj, parse_cag, parse_data

os.makedirs('plots', exist_ok=True)

# prob_id = '609aed2f' # SOI
prob_id = '5e5f5463'
# prob_id = '4510547d' # cheryl's cag
# prob_id = '334000c1' # pred/prey

# 609aed2f/projection-clamp.json   - clamping, in the future
# 609aed2f/projection-clamp-2.json - clamps w/ missing values

print(prob_id)

root       = f"./pam/{prob_id}"
model_path = os.path.join(root, "model.json")
proj_path  = os.path.join(root, "projection.json")

cag  = json.load(open(model_path))
proj = json.load(open(proj_path))

# proj = fix_proj(proj)

# --
# Run API

create_json, res_json = test_api(cag, proj)

print(json.dumps(create_json, indent=2))
# print(json.dumps(res_json, indent=2))




# --
# Plot results

df_cag   = parse_cag(cag['edges'])
df_model = parse_data(cag['nodes'])

for node_obj in res_json['results']:
  node       = node_obj['concept']
  timestamps = [xx['timestamp'] for xx in node_obj['values']]
  values     = np.column_stack([xx['values'] for xx in node_obj['values']])
  
  for v in values:
    _ = plt.plot(timestamps, v, alpha=0.25, c='red')
  
  _ = plt.plot(df_model.timestamp.tail(100), df_model[node].tail(100), marker='o')
  
  show_plot(os.path.join('plots', node.replace('/', '_').replace(' ', '_') + '.png'))

