# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 19:31:13 2018

@author: l00467141
"""

from sklearn.datasets import load_boston

boston = load_boston()

from lightgbm import LGBMRegressor

lgbm = LGBMRegressor(objective = "regression")
lgbm.fit(boston.data, boston.target)

lgbm.booster_.save_model("lightgbm_regression_model.txt")