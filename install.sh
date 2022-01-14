#!/bin/bash

# run.sh

# --
# Setup environment

conda create -y -n sensei_env python=3.8
conda activate sensei_env

pip install -r requirements.txt
conda install uvicorn

pip install requests
pip install pandas
pip install scikit-learn
pip install networkx
