"""
<a href="https://colab.research.google.com/github/souhirbenamor/EPF/blob/main/Mutual_Information.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
https://github.com/krishnaik06/Complete-Feature-Selection/blob/master/4-Information%20gain%20-%20mutual%20information%20In%20Regression.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
import pdb

def mutual_information_feature_selection(df, _feature_file_path, save_df = True):
    """
    df: data frame
    """

    nrows = df.shape[0]
    trainrows = round(nrows * 2/3)
    X=df.drop(df.columns[0], axis = 1)
    y=df[[df.columns[0]]]
    X_train = X[:-trainrows]; X_test = X[-trainrows:]
    y_train = y[:-trainrows]; y_test = y[-trainrows:]

    # determine the mutual information
    mutual_info = mutual_info_regression(X_train.fillna(0), y_train)

    #mutual_info.sort_values(ascending=False).plot.bar(figsize=(15,5))

    ## Selecting the top 30 percentile
    selected_top_columns = SelectPercentile(mutual_info_regression, percentile=30)
    selected_top_columns.fit(X_train.fillna(0), y_train)
    
    feature_colnames = X_train.columns[selected_top_columns.get_support()].tolist()
    select_colnames  = [df.columns[0]] + feature_colnames
    data             = df[select_colnames]

    if save_df:
        # Save dataset with selected feature in another location
        df.to_csv(_feature_file_path)

    print('Performing Feature Selection: Mutual Information')
    print('Selecting Column Names: ', select_colnames)
    
    return data