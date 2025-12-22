import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.model_selection
import sklearn.metrics
import data_load
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import pickle

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# load data with helper
# fourth_downs = data_load.load_fourth_down_data('Data/play_by_play_*.csv')
fourth_downs = pd.read_csv('Data/fourth_down_data.csv')

X = fourth_downs[['ydstogo', 'score_differential', 'game_seconds_remaining', 'posteam_timeouts_remaining', 'yardline_100']]
y = fourth_downs['go']

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.2)

# logistic regression model
# log_reg = lm.LogisticRegression()
# log_reg.fit(X_train, y_train)
# y_pred = log_reg.predict(X_test)

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', lm.LogisticRegression(max_iter=1000))
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)
#print(fourth_downs['go'].mean()) -- naive model would predict only kick since this proportion is ~0.14
print("Accuracy: ", accuracy) # 87.9%, better than naive model (86%), not a great metric for this anyway
auc = sklearn.metrics.roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
print("AUC: ", auc) # 0.84, strong

# save model
os.makedirs("Models", exist_ok=True)

with open("Models/fourth_down_logreg.pkl", "wb") as f:
    pickle.dump(pipe, f)