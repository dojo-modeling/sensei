#!/usr/bin/env python

"""
  utils.py
"""

import json
from sklearn.preprocessing import MinMaxScaler

class FileProgressBar:
  """ file progress bar """
  def __init__(self, fname, max_steps):
    self.fname      = fname
    self._step      = 0
    self._max_steps = max_steps
    
    open(self.fname, 'w').write(json.dumps({"progress" : 0}))
  
  def step(self):
    self._step += 1
    
    progress = round(100 * self._step / self._max_steps)
    open(self.fname, 'w').write(json.dumps({"progress" : progress}))


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
            if node not in df_scaled.columns: continue
            df_scaled[node] = scaler.transform(df_scaled[node].values[:,None]).squeeze()
            
        return df_scaled

    def inverse_transform(self, df_scaled):
        df_unscaled = df_scaled.copy()
        for node, scaler in self.scalers.items():
            df_unscaled[node] = scaler.inverse_transform(df_unscaled[node].values[:,None]).squeeze()
        return df_unscaled
