#!/usr/bin/env python

"""
    engine.py
"""

import os
import json
import arrow
import numpy as np
import pandas as pd
from joblib import dump, load

from .causemos_parsers import parse_cag, parse_data, parse_constraints
from .utils import FileProgressBar, DataFrameScalar
from . import forecast as FF

TRUNCATE_PROJECTION = True # False for debug only ; should be True in prod

# --
# CREATE MODEL

def create_model_output(nodes, df_cag, model):
    out = {
        "status" : "success",
        "nodes"  : [],
        "edges"  : [],
    }

    for node in nodes:
        out['nodes'].append({
            'concept'       : node,
            'scalingFactor' : float(1),
            'scalingBias'   : float(0),
        })

    for edge in df_cag.itertuples():
        
        if edge.src == edge.dst:
            coef = 0.0 # TODO: Fix this -- we don't explicitly use self-loops
        elif edge.dst not in model:
            coef = 0.0
        else:
            coefs = model[edge.dst].get_regression_coefs()
            coef  = float(coefs[coefs.regressor == edge.src].coefficient)

        # QUESTION: Should these coefficients be scaled to match the _transformed_ or _original_ scale of the data?

        out['edges'].append({
            'source'  : edge.src,
            'target'  : edge.dst,
            'weights' : [str(0), str(coef)] # level, trend.  model is y ~ coef * x, so "constant x -> constant y" and "an increase in x -> an increase in y"
        })

    return out


def create_model(cag, model_dirname):
    """
        this call takes CAG w/ everything inside
        outputs model w/ CAG sparsity pattern and optimized weights
    """
    
    if not isinstance(cag, dict):
        cag = cag.dict()
    
    # -
    # get cag / training data
    
    df_cag   = parse_cag(cag['edges'])
    df_model = parse_data(cag['nodes'])
    nodes    = [node['concept'] for node in cag['nodes']]
    periods  = {node['concept']:int(node['period']) for node in cag['nodes']}
    
    progress_bar = FileProgressBar(max_steps=len(nodes), fname=os.path.join(model_dirname, 'progress.json'))
    
    # -
    # Preprocess data

    # min/max scale
    scaler   = DataFrameScalar(nodes)
    scaler   = scaler.fit(df_model)
    df_model = scaler.transform(df_model)
    
    # interpolate regressors
    df_model_interp = FF.interpolate(df_model, df_cag, nodes, method='all')      # temporal interpolation
    
    nodes_interp    = [n for n in nodes if df_model_interp[n].isnull().all()] # graph interpolation
    df_model_interp = FF.interpolate(df_model_interp, df_cag, nodes_interp, method='neibs')

    # -
    # Fit model
    
    model = FF.fit_model(df_model, df_model_interp, df_cag, nodes, periods, shift=True, progress_bar=progress_bar)

    # - 
    # Serialize
    
    state = {
        "df_cag"          : df_cag,
        "df_model"        : df_model,
        "df_model_interp" : df_model_interp,
        "model"           : model,
        "scaler"          : scaler,
        "_meta"           : {
            "nodes"     : nodes,
            # "periods"   : periods,
        }
    }
    dump(state, os.path.join(model_dirname, 'state.pkl'))
    
    # -
    # Return
    
    api_result = create_model_output(nodes, df_cag, model)

    with open(os.path.join(model_dirname, 'create_model_output.json'), 'w') as f:
        f.write(json.dumps(api_result))

    return api_result

# --
# EDIT EDGE

def edit_edge(cag, model_dirname):
    if not isinstance(cag, dict):
        cag = cag.dict()

    # - 
    # Load / edit / save (overwriting previous model)

    state          = load(os.path.join(model_dirname, 'state.pkl'))
    state['model'] = FF.set_user_defined_weights(state['model'], cag)
    dump(state, os.path.join(model_dirname, 'state.pkl'))

    # -
    # Return

    df_cag = state['df_cag']
    model  = state['model']
    nodes  = state['_meta']['nodes']

    api_result = create_model_output(nodes, df_cag, model)

    with open(os.path.join(model_dirname, 'create_model_output.json'), 'w') as f:
        f.write(json.dumps(api_result))

    return api_result

# --
# INVOKE MODEL EXPERIMENT

def invoke_model_experiment_output(df_fut, all_preds):
    out = []
    _date_strs = df_fut._date_str
    for node in all_preds.keys():
        tmp = {
            "concept" : node,
            "values"  : []
        }
        
        for i, _date_str in enumerate(_date_strs):
            tmp['values'].append({
                "timestamp" : int(arrow.get(_date_str).timestamp() * 1000),
                "values"    : list(all_preds[node][i]),
            })
        
        out.append(tmp)
    
    return out


def invoke_model_experiment(model_id, proj, model_dirname, experiment_filename):
    
    if not isinstance(proj, dict):
        proj = proj.dict()
    
    proj_params = proj['experimentParam']

    state = load(os.path.join(model_dirname, 'state.pkl'))
    
    df_cag          = state['df_cag']
    df_model        = state['df_model']
    df_model_interp = state['df_model_interp']
    scaler          = state['scaler']
    model           = state['model']
    nodes           = state['_meta']['nodes']
    
    # interpolate between end of training data and start of projection + create df_fut (w/ clamped values if neccessary)
    df_fut = FF.make_df_fut(
        nodes         = nodes,
        start_time    = int(proj_params['startTime']),
        end_time      = int(proj_params['endTime']),
        num_timesteps = int(proj_params['numTimesteps']),
        last_time     = df_model._date_str.iloc[-1]
    )
    
    # add constraints to df_fut
    proj_params['constraints'] = [c for c in proj_params['constraints'] if len(c['values']) > 0] # hack around empty constraints
    has_constraints = len(proj_params['constraints']) > 0
    if has_constraints:
        df_constraints = parse_constraints(proj_params['constraints'])
        df_constraints = scaler.transform(df_constraints)
        proj_idxs      = df_fut.index[df_fut._proj]
        for node in nodes:
            if node not in df_constraints.columns: continue
            df_fut.loc[proj_idxs[df_constraints.step], node] = df_constraints[node].values
    
    # add "seed" data
    df_fut = pd.concat([df_model_interp.tail(2), df_fut], ignore_index=True).copy()
    
    # forecast
    df_fut, dist_fut = FF.forecast(
        df_fut          = df_fut, 
        model           = model, 
        nodes           = nodes, 
        df_cag          = df_cag, 
        has_constraints = has_constraints,
    )
    
    # drop "seed" data
    df_fut = df_fut.tail(-2)
    
    # truncate to projection date range
    if TRUNCATE_PROJECTION:
        sel = (
            (df_fut._date_str >= arrow.get(proj_params['startTime']).strftime('%Y-%m-%d')) &
            (df_fut._date_str <= arrow.get(proj_params['endTime']).strftime('%Y-%m-%d'))
        ).values
        
        df_fut   = df_fut[sel]
        dist_fut = {k:v[sel] for k,v in dist_fut.items()}
        
    # inverse_scale data
    for node in nodes:
        dist_fut[node] = scaler.scalers[node].inverse_transform(dist_fut[node])
    
    # json format
    api_result = invoke_model_experiment_output(df_fut, dist_fut)
    
    # save
    with open(experiment_filename, 'w') as f:
        f.write(json.dumps(api_result))
    
    return api_result


# --
# CLI Test

if __name__ == '__main__':
    
    model_dirname = 'data/models/'
    
    prob_id = '609aed2f' # SOI
    # prob_id = '4510547d' # cheryl's cag
    # prob_id = '334000c1' # pred/prey

    root    = f'data/pam/{prob_id}'

    # create model
    model_dirname    = os.path.join(root, "model.json")
    model            = json.load(open(model_dirname))
    res_create_model = create_model(model, root)
    print(res_create_model)

    # invoke model expeirment / do projection
    experiment_filename = os.path.join(root, 'test')
    proj_path  = os.path.join(root, "projection.json")
    proj       = h.fix_proj(json.load(open(proj_path)))
    res_invoke_model = invoke_model_experiment(proj, root, experiment_filename)

