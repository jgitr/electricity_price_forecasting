"""
Simplified example for using the DNN model for forecasting prices with daily recalibration
"""

# Author: Jesus Lago

# License: AGPL-3.0 License

from epftoolbox import featureselection
from epftoolbox.models import evaluate_dnn_in_test_dataset, hyperparameter_optimizer, evaluate_lear_in_test_dataset
from epftoolbox.featureselection import perform_recursive_elimination
from epftoolbox.data import read_data
import os
import pdb


# Number of layers in DNN
nlayers = 2

# Market under study. If it not one of the standard ones, the file name
# has to be provided, where the file has to be a csv file
#datasets = ['NP']#['BE', 'DE', 'FR', 'PJM', 'NP']
datasets = ['BE_IMF' + str(x) for x in [0,1,2,3,4,5,6,7,8]]

# Number of years (a year is 364 days) in the test dataset.
years_test = 2

# Boolean that selects whether the validation and training datasets were shuffled when
# performing the hyperparameter optimization. Note that it does not select whether
# shuffling is used for recalibration as for recalibration the validation and the
# training datasets are always shuffled.
shuffle_train = 1

# Boolean that selects whether a data augmentation technique for DNNs is used
data_augmentation = 0

# Boolean that selects whether we start a new recalibration or we restart an existing one
new_recalibration = 1

# Number of years used in the training dataset for recalibration
dnn_calibration_window = 4

# Unique identifier to read the trials file of hyperparameter optimization
experiment_id = 1

# Optional parameters for selecting the test dataset, if either of them is not provided, 
# the test dataset is built using the years_test parameter. They should either be one of
# the date formats existing in python or a string with the following format
# "%d/%m/%Y %H:%M"
begin_test_date = '04/01/2015'
end_test_date = '31/12/2016'

# Set up the paths for saving data (this are the defaults for the library)
path_datasets_folder = os.path.join('.', 'datasets/BE_EEMD')
path_recalibration_folder = os.path.join('.', 'experimental_files')
path_hyperparameter_folder = os.path.join('.', 'experimental_files')

# Number of iterations for hyperparameter optimization
max_evals = 1500

# Boolean that selects whether we start a new hyperparameter optimization or we restart an existing one
new_hyperopt = 1
if not new_hyperopt:
    print('Loading Hyperparameters')


# LEAR paramters (additional)
lear_calibration_window = 364 * 4


for dataset in datasets:
    print('\nCurrent dataset: ', dataset)

    """
    Feature Selection modes:
    ONLY FOR NORDICS

    - RecursiveElimination
    - RandomForest
    - MutualInformation

    If the hyperparamters for the DNN cannot be loaded, then you got to set new_hyperopt = 1
    """
    
    # LEAR
    evaluate_lear_in_test_dataset(path_recalibration_folder=path_recalibration_folder, 
                             path_datasets_folder=path_datasets_folder, dataset=dataset, years_test=years_test, 
                             calibration_window=lear_calibration_window, begin_test_date=begin_test_date, 
                             end_test_date=end_test_date, features=None)
    
    """
    # DNN
    hyperparameter_optimizer(path_datasets_folder=path_datasets_folder, 
                         path_hyperparameters_folder=path_hyperparameter_folder, 
                         new_hyperopt=new_hyperopt, max_evals=max_evals, nlayers=nlayers, dataset=dataset, 
                         years_test=years_test, calibration_window=dnn_calibration_window, 
                         shuffle_train=shuffle_train, data_augmentation=0, experiment_id=experiment_id,
                         begin_test_date=begin_test_date, end_test_date=end_test_date, features=None)
    
    
    evaluate_dnn_in_test_dataset(experiment_id, path_hyperparameter_folder=path_hyperparameter_folder, 
                                path_datasets_folder=path_datasets_folder, shuffle_train=shuffle_train, 
                                path_recalibration_folder=path_recalibration_folder, 
                                nlayers=nlayers, dataset=dataset, years_test=years_test, 
                                data_augmentation=data_augmentation, calibration_window=dnn_calibration_window, 
                                new_recalibration=new_recalibration, begin_test_date=begin_test_date, 
                                end_test_date=end_test_date, features=None)
    """