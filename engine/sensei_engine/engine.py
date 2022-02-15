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

import engine_helpers as h

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
            'scalingBias'   : float(0), # not sure how this is determined or why
        })

    for edge in df_cag.itertuples():
        
        if edge.src == edge.dst:
            coef  = 0.0 # TODO: Fix this -- we don't explicitly use self-loops
        else:
            coefs = model[edge.dst].get_regression_coefs()
            coef  = float(coefs[coefs.regressor == edge.src].coefficient)
        
        out['edges'].append({
            'source'  : edge.src,
            'target'  : edge.dst,
            'weights' : [str(coef), str(0)]
        })

    return out


def create_model(cag, model_dirname):
    """
        this call takes CAG w/ everything inside
        outputs model w/ CAG sparsity pattern and optimized weights
    """
    
    # -
    # get cag / training data
    
    df_cag   = h.parse_cag(cag['edges'])
    df_model = h.parse_data(cag['nodes'])
    nodes    = [node['concept'] for node in cag['nodes']]
    periods  = {node['concept']:12 for node in cag['nodes']}
    
    progress_bar = h.ProgressBar(max_steps=len(nodes), fname=os.path.join(model_dirname, 'progress.json'))
    
    # -
    # Preprocess data

    # min/max scale
    scaler   = h.DataFrameScalar(nodes)
    scaler   = scaler.fit(df_model)
    df_model = scaler.transform(df_model)
    
    # interpolate regressors
    df_model_interp = h.interpolate(df_model, df_cag, nodes, method='all')      # temporal interpolation
    
    nodes_interp    = [n for n in nodes if df_model_interp[n].isnull().all()] # graph interpolation
    df_model_interp = h.interpolate(df_model_interp, df_cag, nodes_interp, method='neibs')

    # -
    # Fit model
    
    model = h.fit_model(df_model, df_model_interp, df_cag, nodes, periods, shift=True, progress_bar=progress_bar)

    # - 
    # Serialize
    
    state = {
        "df_cag"          : df_cag,
        "df_model"        : df_model,
        "df_model_interp" : df_model_interp,
        "model"           : model,
        "scaler"          : scaler,
        "_meta"          : {
            "nodes"     : nodes,
            # "periods"   : periods,
        }
    }
    dump(state, os.path.join(model_dirname, 'state.pkl'))

    with open(os.path.join(model_dirname, 'progress.json'), 'w') as f:
        f.write(json.dumps({"progress" : 100}))
    
    # -
    # Return
    
    api_result = create_model_output(nodes, df_cag, model)
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


def invoke_model_experiment(proj, model_dirname, experiment_filename):
    
    proj_params = proj['experimentParam']

    state = load(os.path.join(model_dirname, 'state.pkl'))
    
    df_cag          = state['df_cag']
    df_model        = state['df_model']
    df_model_interp = state['df_model_interp']
    scaler          = state['scaler']
    model           = state['model']
    nodes           = state['_meta']['nodes']
    
    # interpolate between end of training data and start of projection + create df_fut (w/ clamped values if neccessary)
    df_fut = h.make_df_fut(
        nodes         = nodes,
        start_time    = proj_params['startTime'],
        end_time      = proj_params['endTime'],
        num_timesteps = proj_params['numTimesteps'],
        last_time     = df_model._date_str.iloc[-1]
    )

    df_fut = pd.concat([df_model_interp.tail(2), df_fut], ignore_index=True).copy()
    
    # !! need to verify that these produce ~ the same results
    df_fut, dist_fut = h.forecast(df_fut, model, nodes, df_cag)
    
    # drop "seed" data
    df_fut = df_fut.tail(-2)
    
    # truncate to projection date range
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
    
    # >>
    # SCRATCH: Plotting
    # for node in api_result:
    #     ts     = [xx['timestamp'] for xx in node['values']]
    #     values = np.row_stack([xx['values'] for xx in node['values']]).T
        
    #     for xx in values:
    #         _ = plt.plot(ts, xx, alpha=0.25, c='red')
        
    #     uval = df_model[node['concept']].values
    #     uval = scaler.scalers[node['concept']].inverse_transform(uval[:,None]).squeeze()
    #     _ = plt.plot(df_model.timestamp[-250:], uval[-250:], c='blue')
    #     _ = plt.title(node['concept'])
    #     show_plot(node['concept'].replace(' ', '_') + '-2.png')
    # <<
    
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

