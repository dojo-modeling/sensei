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
    for i, node in enumerate(nodes):
        k  = node['concept']
        ts = [int(vv['timestamp']) for vv in node['values']]
        va = [float(vv['value']) for vv in node['values']]

        if i == 0:
            df = pd.DataFrame(index=ts, data=va, columns=[k])
        else:
            df_ = pd.DataFrame(index=ts, data=va, columns=[k])
            df  = pd.concat([df, df_], axis=1)

    index_date = [datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d') for ts in list(df.index)]
    df['date'] = pd.to_datetime(pd.Series(index_date, index=df.index))
    df         = df.set_index('date').apply(lambda x: x.asfreq(freq='M', method='ffill'))

    if normalize:
        x              = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled       = min_max_scaler.fit_transform(x)
        df             = pd.DataFrame(x_scaled, columns=df.columns, index=df.index)

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

def make_empty_ts_df(start_time, time_steps_in_months, cols):
    start_time = int(start_time)
    end_time   = arrow.get(start_time).shift(months=time_steps_in_months).timestamp()

    tsteps = np.linspace(
        int(start_time),
        int(end_time),
        int(time_steps_in_months),
    ).astype(int)

    index_date = [datetime.fromtimestamp(ts/1000).strftime('%Y-%m-%d') for ts in list(tsteps)]
    return pd.DataFrame(np.nan, index=index_date, columns=cols)


def invoke_model_experiment_output(df_forecast_fut):
    out_data = []
    for c in df_forecast_fut.columns:
        node = {
            "concept" : c,
            "values"  : []
        }
        for timestamp, value in df_forecast_fut[c].iteritems():
            node['values'].append({
                "timestamp" : int(arrow.get(timestamp).timestamp()),
                "values"    : [value] * 100 # TODO: Need individual trajectories, this is just faking
            })

        out_data.append(node)

    return out_data


def invoke_model_experiment(model_id, proj, model_dirname, experiment_filename):
    proj        = proj.dict()
    proj_params = proj['experimentParams']

    df_ts      = pd.read_csv(os.path.join(model_dirname, 'df_ts.csv')).set_index('date')
    df_cag     = pd.read_csv(os.path.join(model_dirname, 'df_cag.csv'))
    df_cag_opt = pd.read_csv(os.path.join(model_dirname, 'df_cag_opt.csv'))

    df_ts_fut  = make_empty_ts_df(
        start_time=proj_params['startTime'],
        time_steps_in_months=proj_params['timeStepsInMonths'],
        cols=df_ts.columns
    )

    # TODO: the dates don't lineup so need to create projection between known values and this df dates
    df_ts_fut   = pd.concat([df_ts, df_ts_fut])
    df_forecast = dyse_rollout(df_cag_opt, df_ts_fut)

    # Format for output
    df_forecast_fut = df_forecast.tail(int(proj_params['timeStepsInMonths']))

    # Save
    res = invoke_model_experiment_output(df_forecast_fut)
    with open(experiment_filename, 'w') as f:
        f.write(json.dumps(res))
