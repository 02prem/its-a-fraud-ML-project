# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from typing import Counter
from sklearn.model_selection import GridSearchCV

train_standard = pd.read_csv('pre-processed_train.csv')
test_standard = pd.read_csv('pre-processed_test.csv')

test_standard.shape, train_standard.shape

X_train = train_standard.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)
y_train = train_standard["isFraud"].to_numpy().astype(np.float64)
X_test = test_standard.drop(axis="columns", labels="isFraud").to_numpy().astype(np.float64)

"""# Re-sampling"""

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

over = RandomOverSampler(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)

pipeline = Pipeline(steps=[('o', over), ('u', under)])
train_X_re, train_y_re = pipeline.fit_resample(X_train, y_train)

"""# XGBoost model"""

import xgboost as xgb

"""param_test1 = {<br>
    'max_depth': [7, 9, 11],<br>
    'min_child_weight': [1, 3]<br>
}<br>

gsearch_1 = GridSearchCV(estimator=xgb_classifier, param_grid=param_test1, scoring='roc_auc', cv=3)

Best parameters:  {'max_depth': 11, 'min_child_weight': 1}<br>

param_test2 = {<br>
    'n_estimators' : [2000, 2500, 3000],<br>
    'max_depth': [8, 10, 12],<br>
    'min_child_weight': [1, 2]<br>
}<br>

gsearch_2 = GridSearchCV(estimator=xgb_classifier, param_grid=param_test2, scoring='roc_auc', cv=3)

Best parameters:  {'max_depth': 12, 'min_child_weight': 1, 'n_estimators': 2000}
"""

xgb_classifier_opt = xgb.XGBClassifier(learning_rate=0.2, n_estimators=2000, tree_method='gpu_hist', max_depth=12)

xgb_classifier_opt.fit(train_X_re, train_y_re)

ypred_test_xgb = xgb_classifier_opt.predict(X_test)
ypred_train_xgb = xgb_classifier_opt.predict(X_train)

print(f1_score(ypred_train_xgb, y_train))

test_results = pd.Series(ypred_test_xgb.astype('int32'), name="isFraud")
test_results.to_csv("submission_1.csv")

ypred_test_xgb_prob = xgb_classifier_opt.predict_proba(X_test)

ypred_test_xgb_prob

prob = []
for i in range(ypred_test_xgb_prob.shape[0]):
    prob.append(ypred_test_xgb_prob[i][1])

prob

test_results = pd.Series(prob, name="isFraud")
test_results.to_csv("submission_1_prob.csv")