#!/usr/bin/env python

"""
  helpers.py
"""

import arrow
import json
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

from orbit.models import DLT

class ProgressBar:
  def __init__(self, fname, max_steps):
    self.fname      = fname
    self._step      = 0
    self._max_steps = max_steps
    
    open(self.fname, 'w').write(json.dumps({"progress" : 0}))
  
  def step(self):
    self._step += 1
    
    progress = round(100 * self._step / self._max_steps)
    open(self.fname, 'w').write(json.dumps({"progress" : progress}))


def make_df_fut(nodes, start_time, end_time, num_timesteps, last_time):
    """ 
        currently only for time steps, 
        need example with clamping to see format to add clamped value
    """
    
    # timesteps for actual projection
    proj_timesteps = arrow.Arrow.range('month', arrow.get(start_time), arrow.get(end_time))
    proj_timesteps = [ts.strftime('%Y-%m-%d') for ts in proj_timesteps]
    assert len(proj_timesteps) == num_timesteps
    
    # timesteps interpolating from last observed data to projection period. ?? is this necessary?
    int_timesteps = arrow.Arrow.range('month', arrow.get(last_time).shift(months=1), arrow.get(start_time).shift(months=-1))
    int_timesteps = [ts.strftime('%Y-%m-%d') for ts in int_timesteps]
    
    df_fut = pd.DataFrame(columns=nodes)
    df_fut['_date_str'] = np.hstack([
        int_timesteps,
        proj_timesteps,
    ])
    df_fut['_date_str'] = pd.to_datetime(df_fut['_date_str'])
    
    return df_fut

def parse_cag(edges):
    """ convert model['edges'] to dataframe """
    cag = []
    for edge in edges:
        src = edge['source']
        dst = edge['target']
        cag.append({
            'src'     : src,
            'dst'     : dst,
            'p_level' : edge['polarity'] * 1,
            'p_trend' : edge['polarity'] * 1,
        })
    
    return pd.DataFrame(cag)


def parse_data(nodes):
    """ convert model['nodes'] to dataframe """
    res = []
    for node in nodes:
        for xx in node['values']:
            res.append({
              "concept"   : node['concept'],
              "timestamp" : xx['timestamp'],
              "value"     : xx['value'] if 'value' in xx else np.mean(xx['values']),
            })
            if 'values' in xx and len(set(xx['values'])) > 1:
                print('!!!')

    res = pd.DataFrame(res)
    res = res.pivot(index="timestamp", columns="concept")

    res = res.value.reset_index()
    res.columns.name = None

    # add date fields
    res['_date']     = res.timestamp.apply(lambda x: arrow.get(x / 1000))
    res['_date_str'] = pd.to_datetime(res._date.apply(lambda x: x.strftime('%Y-%m-%d')))

    return res

def fix_proj(proj):
  """ helper function for cleaning improperly formatted projection parameters """
  
  if 'experimentParams' in proj:
    proj['experimentParam'] = proj.pop('experimentParams')
  
  proj['experimentParam']['numTimesteps'] = proj['experimentParam'].pop('numTimeSteps')
  return proj


class DataFrameScalar:
    """ data frame scaling utility """
    def __init__(self, nodes):
        self.nodes   = nodes
        self.scalers = {node:MinMaxScaler() for node in nodes}

    def fit(self, df):
        for node, scaler in self.scalers.items():
            self.scalers[node] = scaler.fit(df[node].values[:,None])
        return self

    def transform(self, df):
        df_scaled = df.copy()
        for node, scaler in self.scalers.items():
            df_scaled[node] = scaler.transform(df_scaled[node].values[:,None]).squeeze()
        return df_scaled

    def inverse_transform(self, df_scaled):
        df_unscaled = df_scaled.copy()
        for node, scaler in self.scalers.items():
            df_unscaled[node] = scaler.inverse_transform(df_unscaled[node].values[:,None]).squeeze()
        return df_unscaled


def interpolate(df, df_cag, nodes, method='all'):
    assert method in ['all', 'neibs']
    
    df_interp = df.copy()
    if method == 'all': # linear between obs, constant forward / backward
        for node in nodes:
            df_interp[node] = df_interp[node].interpolate(method='linear').bfill().ffill() 

    elif method == 'neibs': # "interpolate" w/ data from neighbors, using simple weighted sum
        for node in nodes:
            in_edges        = df_cag[df_cag.dst == node]
            df_interp[node] = df_interp[in_edges.src].values @ in_edges.p_value.values
        
    return df_interp


def make_graph_input(df, df_interp, df_cag, node, shift=True):
  regressors         = list(set(df_cag.src[df_cag.dst == node]))
  regressors         = [r for r in regressors if r != node]
  df_reg             = df_interp[['_date_str', node] + regressors].copy() # use interpolated data for regressors
  df_reg[node]       = df[node]                                           # use real data for target
  
  if shift:
    df_reg[regressors] = df_reg[regressors].shift(1)                      # use one-step-old regressors - easiest way to handle loops
    df_reg             = df_reg.tail(-1)
  
  return df_reg, regressors


def fit_model(df_train, df_train_interp, df_cag, nodes, periods, shift=True, progress_bar=None):
  models = {}
  for node in nodes:
        
    # make input
    df_reg, regressors = make_graph_input(df_train, df_train_interp, df_cag, node, shift=shift)
    
    # clean target
    if df_reg[node].isnull().all():
      df_reg[node] = df_train_interp[node].copy()     # if all target missing, use "neighbor interpolated version"
    else:
      idx0   = np.where(df_reg[node].notnull())[0][0] # else, truncate to after first non-null observation
      df_reg = df_reg[idx0:]
    
    # fit model
    dlt_params = {
        "response_col"           : node, 
        "regressor_col"          : regressors,
        "date_col"               : '_date_str',
        "estimator"              : 'stan-map',
        "n_bootstrap_draws"      : 200,
        "seed"                   : 123,
        "verbose"                : False,
        "prediction_percentiles" : list(np.arange(10, 95, 5).astype(int)),
    }
    
    try:
      # try w/ correct seasonality .. but sometimes this fails if there's too much missing data...
      orbit_model  = DLT(seasonality=periods[node], **dlt_params)
      models[node] = orbit_model.fit(df=df_reg.tail(1000)) # fit on most recent 1K samples
    except:
      try:
        # if it fails, ignore the seasonality
        print(f'fit_model: error at {node} -- seasonality fail, retrying') # !! How can we work around this
        
        orbit_model  = DLT(**dlt_params)
        models[node] = orbit_model.fit(df=df_reg.tail(1000))
      except:
        print(f'fit_model: error at {node} -- complete fail, skipping') # !! How can we work around this
    
    if progress_bar is not None:
      progress_bar.step()

  return models

def forecast(df_fut, model, nodes, df_cag):
  g = nx.from_pandas_edgelist(df_cag, source='src', target='dst', create_using=nx.DiGraph())
  if nx.is_directed_acyclic_graph(g):
      return _forecast_topology(df_fut, model, list(nx.topological_sort(g)), df_cag)
  else:
      return _forecast_stepwise(df_fut, model, nodes, df_cag)


def _forecast_stepwise(df_fut, model, nodes, df_cag):
  # !! need to handle clamps
  
  dist_fut = defaultdict(lambda: [])
  for idx in trange(2, df_fut.shape[0]):
    for node in nodes:
      
      df_reg, _ = make_graph_input(df_fut.iloc[:idx + 1], df_fut.iloc[:idx + 1], df_cag, node) # !! efficiency
      
      if node in model:
        df_pred = model[node].predict(df=df_reg)
        
        curr_pred      = df_pred.tail(1)
        curr_pred_med  = float(curr_pred.prediction)
        curr_pred_dist = curr_pred[[c for c in curr_pred.columns if 'prediction' in c]].values
        
        df_fut[node].iloc[idx] = curr_pred_med
        dist_fut[node].append(curr_pred_dist)
        
      else:
        # !! failed to train a model -- fall back to just carrying forward values w/ no variance.
        val                    = float(df_reg[node][df_reg[node].notnull()].iloc[-1])
        df_fut[node].iloc[idx] = val
        dist_fut[node].append(np.ones(17) * val) # number of percentiles

  dist_fut = {k:np.row_stack(v) for k,v in dist_fut.items()}
  return df_fut, dist_fut


def _forecast_topology(df_fut, model, nodes, df_cag):
  # !! need to handle clamps
  
  dist_fut = {}
  for node in nodes:
    df_reg, _ = make_graph_input(df_fut, df_fut, df_cag, node)
    
    if node in model:
      df_pred   = model[node].predict(df=df_reg)
      
      df_pred   = df_pred.tail(-1)
      pred_med  = df_pred.prediction.values
      pred_dist = df_pred[[c for c in df_pred.columns if 'prediction' in c]].values
      
      df_fut[node].iloc[2:] = pred_med
      dist_fut[node] = pred_dist
    else:
      val            = float(df_reg[node][df_reg[node].notnull()].iloc[-1])
      df_fut[node]   = val
      dist_fut[node] = np.ones((df_fut.shape[0], 17)) * val
  
  return df_fut, dist_fut
