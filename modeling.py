import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection
import sklearn.metrics
import data_load

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# load data with helper
fourth_downs = data_load.load_fourth_down_data('Data/play_by_play_*.csv')

X = fourth_downs[['ydstogo', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining']]
y = fourth_downs['go']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# logistic regression model
log_reg = lm.LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
#print(fourth_downs['go'].mean()) -- naive model would predict only kick since this proportion is ~0.14
print("Accuracy: ", accuracy) # 87.7%, better than naive model (86%), not a great metric for this anyway
auc = sklearn.metrics.roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
print("AUC: ", auc) # 0.83, strong