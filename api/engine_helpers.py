import pandas as pd
import numpy as np
from datetime import datetime
import networkx as nx
from scipy.optimize import minimize
from sklearn import preprocessing

def dyse_rollout(cag, data):
    n_node     = len(data.columns)
    key2idx    = {k:i for i,k in enumerate(data.columns)}
    data_index = data.index
    data       = data.values

    # transition matrices
    l_trans = np.zeros((n_node, n_node))
    t_trans = np.zeros((n_node, n_node))
    for e in cag.itertuples():
        l_trans[key2idx[e.dst], key2idx[e.src]] = e.p_level
        t_trans[key2idx[e.dst], key2idx[e.src]] = e.p_trend

    # init + propagate
    levels = data[0]
    trends = np.zeros_like(data[0])
    out    = [levels]
    for i in range(1, len(data)):
        new_levels = levels + l_trans @ levels + t_trans @ trends

        sel = ~np.isnan(data[i])
        new_levels[sel] = data[i, sel]

        # clip 
        #new_levels = new_levels.clip(min=0, max=n_levels)

        # update state
        trends = new_levels - levels
        levels = new_levels
        out.append(levels)
    
    out         = pd.DataFrame(out)
    out.index   = data_index
    out.columns = list(key2idx.keys())
    
    return out
 

def dyse_optimize(cag, data, obs):
    def _opt_wrapper(params, cag):
        cag = cag.copy()
        cag.p_level = params[:cag.shape[0]]
        cag.p_trend = params[cag.shape[0]:]
        pred = dyse_rollout(cag, data)
        loss = ((pred.values - obs.values) ** 2).mean()
        return loss

    cag_opt = cag.copy()
    cag_opt['p_level'] = np.nan
    cag_opt['p_trend'] = np.nan
    x0 = np.zeros(2 * cag.shape[0])
    x  = minimize(_opt_wrapper, x0, args=(cag_opt,), options={'disp': True}).x

    cag_opt.p_level = x[:cag.shape[0]]
    cag_opt.p_trend = x[cag.shape[0]:]
    return cag_opt

