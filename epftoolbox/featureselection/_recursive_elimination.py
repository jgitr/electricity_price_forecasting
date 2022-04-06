"""
<a href="https://colab.research.google.com/github/souhirbenamor/EPF/blob/main/Recursive_feature_Elimination.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
https://github.com/MachineLearnia/Python-Machine-Learning/blob/master/23%20-%20Sklearn%20Feature%20Selection.**ipynb**
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFECV
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectFromModel

def perform_recursive_elimination(df, _feature_file_path, save_df = True):
    """
    df: data frame
    """

    X=df.drop(df.columns[0], axis = 1)
    y=df[[df.columns[0]]]

    feature_names = list(X.columns.values)
    print('feature names: ', feature_names)

    # Variance Threshold
    X.var(axis=0)
    selector = VarianceThreshold(threshold=0.2)
    selector.fit(X)
    selector.get_support()
    np.array(feature_names)[selector.get_support()]
    selector.variances_

    # 2. SelectKBest
    y=y.astype('int')
    chi2(X, y)
    selector = SelectKBest(f_classif, k=2)
    selector.fit(X, y)
    selector.scores_
    np.array(feature_names)[selector.get_support()]

    # 3. Recursive Feature Elemination
    selector = RFECV(SGDClassifier(random_state=0), step=1, min_features_to_select=2, cv=5)
    selector.fit(X, y)
    print(selector.ranking_)
    print(selector.grid_scores_)

    np.array(feature_names)[selector.get_support()]

    # 4. Select From linear_model
    selector = SelectFromModel(SGDClassifier(random_state=0), threshold='mean')
    selector.fit(X, y)
    coefs = selector.estimator_.coef_

    print('Selected Coefficients: ', coefs)

    # Column names of selected variables
    feature_colnames = np.array(feature_names)[selector.get_support()].tolist()
    
    select_colnames  = [df.columns[0]] + feature_colnames
    data             = df[select_colnames]

    if save_df:
        # Save dataset with selected feature in another location
        df.to_csv(_feature_file_path)

    print('Performing Feature Selection: Recursive Elimination')
    print('Selecting Column Names: ', select_colnames)
    
    return data