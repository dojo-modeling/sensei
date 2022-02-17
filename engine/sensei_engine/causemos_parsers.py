#!/usr/bin/env python

"""
  causemos_parsers.py
"""

import arrow
import numpy as np
import pandas as pd

def fix_proj(proj):
  """ helper function for cleaning improperly formatted projection parameters """
  
  if 'experimentParams' in proj:
    proj['experimentParam'] = proj.pop('experimentParams')
  
  proj['experimentParam']['numTimesteps'] = proj['experimentParam'].pop('numTimeSteps')
  return proj


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


def parse_constraints(nodes):
    """ convert model['nodes'] to dataframe """
    res = []
    for node in nodes:
        for xx in node['values']:
            res.append({
              "concept" : node['concept'],
              "step"    : int(xx['step']),
              "value"   : xx['value'],
            })
    
    res = pd.DataFrame(res)
    res = res.pivot(index="step", columns="concept")
    
    res = res.value.reset_index()
    res.columns.name = None
    return res
