"""
<a href="https://colab.research.google.com/github/souhirbenamor/EPF/blob/main/Feature_selection_with_Random_Forest_regressor.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

https://www.yourdatateacher.com/2021/10/11/feature-selection-with-random-forest/
"""

import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import pdb

def random_forest_feature_selection(df, _feature_file_path, save_df = True):
    """
    df: data frame
    """
    nrows = df.shape[0]
    trainrows = round(nrows * 2/3)

    X=df.drop(df.columns[0], axis = 1)
    y=df[[df.columns[0]]]

    X_train = X[:-trainrows]; X_test = X[-trainrows:]
    y_train = y[:-trainrows]; y_test = y[-trainrows:]

    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    rf = RandomForestRegressor(random_state=0)

    rf.fit(X_train,y_train)

    features = list(X.columns.values)

    print('feature importance:', rf.feature_importances_)

    f_i = list(zip(features,rf.feature_importances_))
    f_i.sort(key = lambda x : x[1])
    #plt.barh([x[0] for x in f_i],[x[1] for x in f_i])
    #plt.show()

    rfe = RFECV(rf,cv=5,scoring="neg_mean_squared_error")

    rfe.fit(X_train,y_train)

    feature_colnames = np.array(features)[rfe.get_support()].tolist()

    select_colnames  = [df.columns[0]] + feature_colnames
    data             = df[select_colnames]

    if save_df:
        # Save dataset with selected feature in another location
        df.to_csv(_feature_file_path)

    print('Performing Feature Selection: Random Forest')
    print('Selecting Column Names: ', select_colnames)
    
    return data


