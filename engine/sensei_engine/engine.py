# import json
import os
import arrow
import json
import numpy as np
import pandas as pd
from datetime import datetime
# import matplotlib.pyplot as plt
# import networkx as nx
# from itertools import cycle

from .engine_helpers import dyse_optimize, dyse_rollout

# --
# Helpers

def get_ts(nodes, normalize=True):
    df = []
    for node in nodes:
        k  = node['concept']
        ts = [int(vv['timestamp']) for vv in node['values']]
        va = [float(vv['value']) for vv in node['values']]

        tmp = pd.DataFrame(index=ts, data=va, columns=[k])
        df.append(tmp)
    
    df = pd.concat(df, axis=1).sort_index()

    index_date = [arrow.get(xx // 1000).strftime('%Y-%m-%d') for xx in list(df.index)]
    df['date'] = pd.to_datetime(pd.Series(index_date, index=df.index))
    df         = df.set_index('date').apply(lambda x: x.asfreq(freq='M', method='ffill'))

    if normalize:
        print('!! NOT IMPLEMENTED')
        # print('normalize')
        # x              = df.values #returns a numpy array
        # min_max_scaler = preprocessing.MinMaxScaler()
        # x_scaled       = min_max_scaler.fit_transform(x)
        # df             = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)

    return df.sort_index()


def clean_ts(df_ts):
    df_ts = df_ts.interpolate(limit_direction='both')
    for c in df_ts.columns:
        if df_ts[c].isnull().all():
            df_ts[c] = 0

    return df_ts

def get_cag(edges):
    cag = []
    for edge in edges:
        dst = edge['source']
        src = edge['target']
        cag.append({
            'src'     : src,
            'dst'     : dst,
            'p_level' : 0.01,
            'p_trend' : 0.01
        })

    return pd.DataFrame(cag)

# --
# API

def create_model_output(payload, df_cag_opt):
    out = {
        "status" : "success",
        "nodes"  : [],
        "edges"  : [],
    }

    for node in payload['nodes']:
        out['nodes'].append({
            'concept'       : node['concept'],
            'scalingFactor' : float(1),
            'scalingBias'   : float(0), # not sure how this is determined or why
        })

    for edge in df_cag_opt.itertuples():
        out['edges'].append({
            'source'  : edge.src,
            'target'  : edge.dst,
            'weights' : [
                str(edge.p_level),
                str(edge.p_trend),
            ]
        })

    return out

# --
# API Hooks

def create_model(cag, model_dirname):
    """
        this call takes CAG w/ everything inside
        outputs model w/ CAG sparsity pattern and optimized weights
    """

    with open(os.path.join(model_dirname, 'progress.json'), 'w') as f:
        f.write(json.dumps({"progress" : 0}))

    cag = cag.dict()

    # time series data
    df_ts = get_ts(cag['nodes'], normalize=False)
    df_ts = clean_ts(df_ts)

    # cag graph data
    df_cag = get_cag(cag['edges'])

    # run model
    obs           = df_ts.copy()
    data          = df_ts.copy()
    data.iloc[1:] = np.nan # have to keep first row
    df_cag_opt    = dyse_optimize(df_cag, data, obs)

    # create output uncharted would like (to mimic old api calls)
    output = create_model_output(cag, df_cag_opt)

    # save
    df_ts.to_csv(os.path.join(model_dirname, 'df_ts.csv'))
    df_cag.to_csv(os.path.join(model_dirname, 'df_cag.csv'))
    df_cag_opt.to_csv(os.path.join(model_dirname, 'df_cag_opt.csv'))

    with open(os.path.join(model_dirname, 'create_model_output.json'), 'w') as f:
        f.write(json.dumps(output))

    with open(os.path.join(model_dirname, 'progress.json'), 'w') as f:
        f.write(json.dumps({"progress" : 100}))

    return output

# --

def make_empty_ts_df(start_time, end_time, time_steps_in_months, cols):
    timesteps = arrow.Arrow.range('month', arrow.get(start_time), arrow.get(end_time))
    timesteps = [ts.strftime('%Y-%m-%d') for ts in timesteps]
    assert len(timesteps) == time_steps_in_months
    return pd.DataFrame(np.nan, index=timesteps, columns=cols)


def invoke_model_experiment_output(df_forecast_fut):
    out_data = []
    df_forecast_fut = df_forecast_fut.sort_index()
    for c in df_forecast_fut.columns:
        node = {
            "concept" : c,
            "values"  : []
        }
        for timestamp, value in df_forecast_fut[c].iteritems():
            node['values'].append({
                "timestamp" : int(arrow.get(timestamp).timestamp() * 1000),
                "values"    : [value] * 100 # TODO: Need individual trajectories, this is just faking
            })

        out_data.append(node)

    return out_data


def invoke_model_experiment(model_id, proj, model_dirname, experiment_filename):
    proj        = proj.dict()
    proj_params = proj['experimentParam']

    df_ts      = pd.read_csv(os.path.join(model_dirname, 'df_ts.csv')).set_index('date')
    df_cag_opt = pd.read_csv(os.path.join(model_dirname, 'df_cag_opt.csv'))

    df_ts_fut  = make_empty_ts_df(
        start_time=proj_params['startTime'],
        end_time=proj_params['endTime'],
        time_steps_in_months=proj_params['numTimesteps'],
        cols=df_ts.columns
    )

    # TODO: the dates don't lineup so need to create projection between known values and this df dates
    df_ts_fut   = pd.concat([df_ts, df_ts_fut])
    df_forecast = dyse_rollout(df_cag_opt, df_ts_fut)
    
    # Format for output
    sel = (
        (df_forecast.index >= arrow.get(proj_params['startTime']).strftime('%Y-%m-%d')) &
        (df_forecast.index <= arrow.get(proj_params['endTime']).strftime('%Y-%m-%d'))
    )
    assert sel.sum() == proj_params['numTimesteps']
    df_forecast_fut = df_forecast[sel]
    
    # Save
    res = invoke_model_experiment_output(df_forecast_fut)
    with open(experiment_filename, 'w') as f:
        f.write(json.dumps(res))
