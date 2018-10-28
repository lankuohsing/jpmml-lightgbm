# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 21:21:47 2018

@author: l00467141
"""


import pandas as pd
from lightgbm import LGBMClassifier
iris_df = pd.read_csv("./iris.csv")
d_x = iris_df.iloc[:, 1:5].values
d_y = iris_df.iloc[:, 5].values
model = LGBMClassifier(
    boosting_type='gbdt', objective="multiclass", nthread=8, seed=42)
model.n_classes =3
model.fit(d_x,d_y,feature_name=iris_df.columns.tolist()[1:-1])
model.booster_.save_model("lightgbm_iris_model.txt")

