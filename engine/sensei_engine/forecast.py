#!/usr/bin/env python

"""
  forecast.py
"""

import arrow
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import trange, tqdm
from collections import defaultdict

from orbit.models import DLT

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

def infer_timestep_size(start_time, end_time, num_timesteps):
  MS_YEAR  = (1000 * 60 * 60 * 24 * 365)
  MS_MONTH = MS_YEAR // 12

  timestep_ms = (end_time - start_time) // (num_timesteps - 1)
  err_year    = abs(timestep_ms / MS_YEAR - 1)
  err_month   = abs(timestep_ms / MS_MONTH - 1)
  if err_year < err_month:
    return 'year'
  else:
    return 'month'

  
def make_df_fut(nodes, start_time, end_time, num_timesteps, last_time):
    """ 
        currently only for time steps, 
        need example with clamping to see format to add clamped value
    """
    timestep_size = infer_timestep_size(start_time, end_time, num_timesteps)
    print(f'make_df_fut: timestep_size={timestep_size}')
    
    # timesteps for actual projection
    proj_timesteps = arrow.Arrow.range(timestep_size, arrow.get(start_time), arrow.get(end_time))
    proj_timesteps = [ts.strftime('%Y-%m-%d') for ts in proj_timesteps]
    assert len(proj_timesteps) == num_timesteps
    
    # timesteps interpolating from last observed data to projection period. ?? is this necessary?
    int_timesteps   = arrow.Arrow.range(
      timestep_size, 
      arrow.get(last_time).shift(**{f'{timestep_size}s' : 1}), 
      arrow.get(start_time).shift(**{f'{timestep_size}s' : -1}), 
    )
    int_timesteps   = [ts.strftime('%Y-%m-%d') for ts in int_timesteps]
    
    df_fut = pd.DataFrame(columns=nodes)
    df_fut['_proj'] = np.hstack([
      0 * np.ones(len(int_timesteps)),
      1 * np.ones(len(proj_timesteps))
    ]).astype(bool)
    
    df_fut['_date_str'] = np.hstack([
        int_timesteps,
        proj_timesteps,
    ])
    df_fut['_date_str'] = pd.to_datetime(df_fut['_date_str'])
    
    for node in nodes:
      df_fut[node] = df_fut[node].astype(np.float64)
    
    return df_fut

  
def make_df_reg(df, df_interp, df_cag, node, shift=True):
  df_                = df_cag.loc[df_cag.dst == node]
  regressors         = list(set(df_.src))
  regressors         = [r for r in regressors if r != node]
  sign_dict          = {-1: '-', 0: '=', 1: '+'}
  sign               = [sign_dict[int(df_.p_level.loc[df_cag.src == r])] for r in regressors]
  df_reg             = df_interp[['_date_str', node] + regressors].copy() # use interpolated data for regressors
  df_reg[node]       = df[node]                                           # use real data for target

  if shift:
    df_reg[regressors] = df_reg[regressors].shift(1)                      # use one-step-old regressors - easiest way to handle loops
    df_reg             = df_reg.tail(-1)

  return df_reg, regressors, sign


def fit_model(df_train, df_train_interp, df_cag, nodes, periods, shift=True, progress_bar=None):
  models = {}
  for node in nodes:
    print('-' * 100)
    print(f'fit_model: {node}')
    
    # make input
    df_reg, regressors, sign = make_df_reg(df_train, df_train_interp, df_cag, node, shift=shift)
    
    # clean target
    if df_reg[node].isnull().all():
      df_reg[node] = df_train_interp[node].copy()     # if all target missing, use "neighbor interpolated version"
    else:
      idx0   = np.where(df_reg[node].notnull())[0][0] # else, truncate to after first non-null observation
      df_reg = df_reg[idx0:]
    
    df_reg = df_reg.tail(1000) # fit on most recent 1K samples
    
    print(node, regressors)
    # fit model
    dlt_params = {
        "response_col"           : node, 
        "regressor_col"          : regressors,
        "regressor_sign"         : sign,
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
      models[node] = orbit_model.fit(df=df_reg) 
    except Exception as e0:
      try:
        print(f'fit_model: {node} | seasonality fail | retrying w/o seasonality') # !! How can we work around this
        orbit_model  = DLT(**dlt_params)
        models[node] = orbit_model.fit(df=df_reg)
      except:
        print(f'fit_model: {node} | complete fail | skipping | {e0.args}') # !! How can we work around this
    
    if progress_bar is not None:
      progress_bar.step()

  return models


def _forecast_stepwise(model, df_fut, df_cag, nodes):
  print('_forecast_stepwise')

  dist_fut = defaultdict(lambda: [])
  for idx in trange(2, df_fut.shape[0]):
    for node in nodes:
      
      df_reg, _, _  = make_df_reg(df_fut.iloc[:idx + 1], df_fut.iloc[:idx + 1], df_cag, node) # !! efficiency
      is_clamped = not np.isnan(df_fut.loc[idx, node])
      
      if is_clamped:
        val = df_fut.loc[idx, node]
        dist_fut[node].append(np.ones(17) * val)
      
      elif node in model:
        df_pred = model[node].predict(df=df_reg)
        
        curr_pred      = df_pred.tail(1)
        curr_pred_med  = float(curr_pred.prediction)
        curr_pred_dist = curr_pred[[c for c in curr_pred.columns if 'prediction' in c]].values
        
        df_fut.loc[idx, node] = curr_pred_med
        dist_fut[node].append(curr_pred_dist)
        
      else:
        # !! failed to train a model -- fall back to just carrying forward values w/ no variance.
        val                   = float(df_reg[node][df_reg[node].notnull()].iloc[-1])
        df_fut.loc[idx, node] = val
        dist_fut[node].append(np.ones(17) * val) # number of percentiles

  dist_fut = {k:np.row_stack(v) for k,v in dist_fut.items()}
  return df_fut, dist_fut


def _forecast_topology(model, df_fut, df_cag, nodes):
  # !! can this handle clamps?  not obvious IMO ...
  
  print('_forecast_topology')
  
  dist_fut = {}
  for node in tqdm(nodes):
    df_reg, _, _ = make_df_reg(df_fut, df_fut, df_cag, node)
    
    if node in model:
      df_pred   = model[node].predict(df=df_reg)
      
      df_pred   = df_pred.tail(-1)
      pred_med  = df_pred.prediction.values
      pred_dist = df_pred[[c for c in df_pred.columns if 'prediction' in c]].values
      
      df_fut.loc[2:, node] = pred_med
      dist_fut[node]       = pred_dist
    else:
      val            = float(df_reg[node][df_reg[node].notnull()].iloc[-1])
      df_fut[node]   = val
      dist_fut[node] = np.ones((df_fut.shape[0], 17)) * val
  
  return df_fut, dist_fut


def forecast(model, df_fut, df_cag, nodes, has_constraints):
  """ 
    forecasting function 
    
    forecasting in topological order is substantially faster, but if the graph
      - has cycles
      - has clamps (for not all of the points)
    then we have to do stepwise.
  """
  g = nx.from_pandas_edgelist(df_cag, source='src', target='dst', create_using=nx.DiGraph())
  if (not nx.is_directed_acyclic_graph(g)) or (has_constraints):
      return _forecast_stepwise(model, df_fut, df_cag, nodes)
  else:
      return _forecast_topology(model, df_fut, df_cag, list(nx.topological_sort(g)))
