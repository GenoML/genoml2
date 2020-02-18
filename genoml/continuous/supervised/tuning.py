# Importing the necessary packages 
import argparse
import sys
import h5py
import pandas as pd
import numpy as np
from time import time

# Import the necessary internal GenoML packages 
from genoml.continuous.supervised import train

# Define the tune class 
class tune():
    def __init__(self, df, run_prefix, max_iter, cv_count):
        self.run_prefix = run_prefix
        self.max_iter = max_iter
        self.cv_count = cv_count
       
        self.y_tune = df.PHENO
        self.X_tune = df.drop(columns=['PHENO'])
        self.IDs_tune = self.X_tune.ID
        self.X_tune = self.X_tune.drop(columns=['ID'])

        best_algo_name_in = run_prefix + '.best_algorithm.txt'
        best_algo_df = pd.read_csv(best_algo_name_in, header=None, index_col=False)
        self.best_algo = str(best_algo_df.iloc[0,0])

        self.algorithms = [
            LinearRegression(),
            RandomForestRegressor(),
            AdaBoostRegressor(),
            GradientBoostingRegressor(),
            SGDRegressor(),
            SVR(),
            MLPRegressor(),
            KNeighborsRegressor(),
            BaggingRegressor(),
            XGBRegressor()
        ]

        # Initialize a few variables we will be using later 
        self.log_table = None
        self.best_algo_name_in = None
        self.best_algo_df = None
        self.hyperparameters = None
        self.scoring_metric = None
        self.cv_tuned = None 
        self.cv_baseline = None 
        self.algo = None
        self.searchCVResults = None
        self.rand_search = None
        self.algo_tuned = None
        self.tune_out = None

    def select_tuning_parameters(self):
        best_algo = self.best_algo

        if best_algo == 'LinearRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif  best_algo == 'SGDRegressor':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
            algo = getattr(sklearn.ensemble, best_algo)()

        elif best_algo == 'SVR':
            algo = getattr(sklearn.svm, best_algo)(gamma='auto')

        elif best_algo == 'MLPRegressor':
            algo = getattr(sklearn.neural_network, best_algo)()

        elif best_algo == 'XGBRegressor':
            algo = getattr(xgboost, best_algo)()

        elif best_algo == 'KNeighborsRegressor':
            algo = getattr(sklearn.neighbors, best_algo)()
        
        self.algo = algo 

        if best_algo == 'LinearRegression':
            hyperparameters = {"penalty": ["l1", "l2"], "C": sp_randint(1, 10)}
            scoring_metric = make_scorer(explained_variance_score)

        elif  best_algo == 'SGDRegressor':
            hyperparameters = {'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], "learning_rate": ["constant", "optimal", "invscaling", "adaptive"]}
            scoring_metric = make_scorer(explained_variance_score)

        elif (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
            hyperparameters = {"n_estimators": sp_randint(1, 1000)}
            scoring_metric = make_scorer(explained_variance_score)

        elif best_algo == 'SVR':
            hyperparameters = {"kernel": ["linear", "poly", "rbf", "sigmoid"], "C": sp_randint(1, 10)}
            scoring_metric = make_scorer(explained_variance_score)
            
        elif best_algo == 'MLPRegressor':
            hyperparameters = {"alpha": sp_randfloat(0,1), "learning_rate": ['constant', 'invscaling', 'adaptive']}
            scoring_metric = make_scorer(explained_variance_score)

        elif best_algo == 'XGBRegressor':
            hyperparameters = {"max_depth": sp_randint(1, 100), "learning_rate": sp_randfloat(0,1), "n_estimators": sp_randint(1, 100), "gamma": sp_randfloat(0,1)}
            scoring_metric = make_scorer(explained_variance_score)

        elif best_algo == 'KNeighborsRegressor':
            hyperparameters = {"leaf_size": sp_randint(1, 100), "n_neighbors": sp_randint(1, 10)}
            scoring_metric = make_scorer(explained_variance_score)

        self.hyperparameters = hyperparameters
        self.scoring_metric = scoring_metric

        return algo, hyperparameters, scoring_metric

