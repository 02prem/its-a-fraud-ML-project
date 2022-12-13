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

"""# Naive bayes model"""

from sklearn.naive_bayes import GaussianNB

# params_nb = {
#             'var_smoothing': [1e-9, 1e-6, 1e-12]
#             }

# gaussian_nb_grid = GridSearchCV(GaussianNB(), param_grid=params_nb, n_jobs=-1, cv=10, verbose=1)

# gaussian_nb_grid.fit(train_X_re, train_y_re)

# print('Best Parameters : {}\n'.format(gaussian_nb_grid.best_params_))

"""Best Parameters : {'var_smoothing': 1e-12}"""

nb_model = GaussianNB(var_smoothing=1e-12)

nb_model.fit(train_X_re, train_y_re)

ypred_nb = nb_model.predict(X_test)

test_results_1 = pd.Series(ypred_nb.astype('int32'), name="isFraud")
test_results_1.to_csv("submission_1.csv")