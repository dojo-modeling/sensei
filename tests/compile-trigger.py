#!/usr/bin/env python

"""
   compile-trigger.py
"""

import sys
print("-" * 50, file=sys.stderr)
print("compile-trigger.py: start", file=sys.stderr)

import arrow
import numpy as np
import pandas as pd
from orbit.models import DLT

_date_str = [d.strftime('%Y-%m-%d') for d in arrow.Arrow.range(
  frame='month',
  start=arrow.get('2021-01-01'),
  end=arrow.get('2021-12-01'),
)]

df = pd.DataFrame({
  "x0"        : np.random.uniform(12),
  "x1"        : np.random.uniform(12),
  "y"         : np.random.uniform(12),
  "_date_str" : _date_str
})


dlt_params = {
    "response_col"           : "y", 
    "regressor_col"          : ["x0", "x1"],
    "date_col"               : '_date_str',
    "estimator"              : 'stan-map',
    "n_bootstrap_draws"      : 200,
    "seed"                   : 123,
    "verbose"                : False,
    "prediction_percentiles" : list(np.arange(5, 95, 1).astype(int)),
}

model = DLT(seasonality=6, **dlt_params)
model = model.fit(df=df)
_ = model.predict(df)

print("compile-trigger.py: done", file=sys.stderr)
print("-" * 50, file=sys.stderr)