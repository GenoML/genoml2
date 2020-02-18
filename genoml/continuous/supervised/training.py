# Importing the necessary packages 
import argparse
import sys
import xgboost
import sklearn
import pandas as pd
import numpy as np
import time
import seaborn as sns
import statsmodels.formula.api as sm
from joblib import dump, load

# Import the necessary ML packages 
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score, mean_squared_error, median_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.feature_selection import RFE

# Define the train class 
class train:
    def __init__(self, df, run_prefix):
        # Prepping the data
        # Drop the PHENO column first prior to splitting 
        y = df.PHENO
        X = df.drop(columns=['PHENO'])
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # 70:30
        IDs_train = X_train.ID
        IDs_test = X_test.ID
        X_train = X_train.drop(columns=['ID'])
        X_test = X_test.drop(columns=['ID'])
        
        # Saving the prepped data the other classes will need
        self.df = df
        self.run_prefix = run_prefix
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.IDs_train = IDs_train
        self.IDs_test = IDs_test

        # Where the results will be stored 
        self.log_table = None
        self.best_algo = None
        self.algo = None
        self.rfe_df = None

        # The algorithms we will use 
        self.algorithms = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10),
        AdaBoostRegressor(),
        GradientBoostingRegressor(),
        SGDRegressor(),
        SVR(gamma='auto'),
        MLPRegressor(),
        KNeighborsRegressor(),
        BaggingRegressor(),
        XGBRegressor()
        ]        

    # Report and data summary you want 
    def summary(self):
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(self.df.describe())
        print("#"*70)

    # Compete the algorithms 
    def compete(self, verbose=False):
        log_cols=["Algorithm", "Explained_variance_score", "Mean_squared_error", "Median_absolute_error", "R2_score", "Runtime_Seconds"]
        log_table = pd.DataFrame(columns=log_cols)

        for algo in self.algorithms:
            start_time = time.time()

            algo.fit(self.X_train, self.y_train)
            name = algo.__class__.__name__

            print("")
            print("#"*70)
            print("")
            print(name)

            test_predictions = algo.predict(self.X_test)
            test_predictions = test_predictions
            evs = explained_variance_score(self.y_test, test_predictions)
            print("Explained Variance Score: {:.4}".format(evs))

            test_predictions = algo.predict(self.X_test)
            test_predictions = test_predictions
            mse = mean_squared_error(self.y_test, test_predictions)
            print("Mean Squared Error: {:.4}".format(mse))

            test_predictions = algo.predict(self.X_test)
            test_predictions = test_predictions
            mae = median_absolute_error(self.y_test, test_predictions)
            print("Median Absolute Error: {:.4}".format(mae))

            test_predictions = algo.predict(self.X_test)
            test_predictions = test_predictions
            r2s = r2_score(self.y_test, test_predictions)
            print("R^2 Score: {:.4}".format(r2s))

            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("Runtime in seconds: {:.4}".format(elapsed_time))

            log_entry = pd.DataFrame([[name, evs, mse, mae, r2s, elapsed_time]], columns=log_cols)
            log_table = log_table.append(log_entry)

        print("#"*70)
        print("")

        self.log_table = log_table
        return log_table 

    def results(self):
        best_performing_summary = self.log_table[self.log_table.Explained_variance_score == self.log_table.Explained_variance_score.max()]
        best_algo = best_performing_summary.at[0,'Algorithm']
        
        self.best_algo = best_algo

        return best_algo

    def export_model(self):
        best_algo = self.best_algo
        if best_algo == 'LinearRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif  best_algo == 'SGDRegressor':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif (best_algo == 'RandomForestRegressor') or (best_algo == 'AdaBoostRegressor') or (best_algo == 'GradientBoostingRegressor') or  (best_algo == 'BaggingRegressor'):
            algo = getattr(sklearn.ensemble, best_algo)()

        elif best_algo == 'SVR':
            algo = getattr(sklearn.svm, best_algo)()

        elif best_algo == 'MLPRegressor':
            algo = getattr(sklearn.neural_network, best_algo)()

        elif best_algo == 'XGBRegressor':
            algo = getattr(xgboost, best_algo)()

        elif best_algo == 'KNeighborsRegressor':
            algo = getattr(sklearn.neighbors, best_algo)()

        algo.fit(self.X_train, self.y_train)
        name = algo.__class__.__name__

        print("...remember, there are occasionally slight fluctuations in model performance on the same withheld samples...")
        print("#"*70)
        print(name)

        test_predictions = algo.predict(self.X_test)
        test_predictions = test_predictions
        evs = explained_variance_score(self.y_test, test_predictions)
        print("Explained Variance Score: {:.4}".format(evs))

        test_predictions = algo.predict(self.X_test)
        test_predictions = test_predictions
        mse = mean_squared_error(self.y_test, test_predictions)
        print("Mean Squared Error: {:.4}".format(mse))

        test_predictions = algo.predict(self.X_test)
        test_predictions = test_predictions
        mae = median_absolute_error(self.y_test, test_predictions)
        print("Median absolut error: {:.4}".format(mae))

        test_predictions = algo.predict(self.X_test)
        test_predictions = test_predictions
        r2s = r2_score(self.y_test, test_predictions)
        print("R^2 score: {:.4}".format(r2s))

        ### Save it using joblib
        algo_out = self.run_prefix + '.trainedModel.joblib'
        dump(algo, algo_out)

        print("#"*70)
        print(f"... this model has been saved as {algo_out} for later use and can be found in your working directory.")

        self.algo = algo

        return algo 

    def export_predictions(self):
        test_predicted_values = algo.predict(self.X_test)
        test_predicted_values_df = pd.DataFrame(test_predicted_values)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        test_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
        test_out = test_out.drop(columns=["INDEX"])

        test_outfile = self.run_prefix + '.trainedModel_withheldSample_Predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the withheld test data that has been exported as {test_outfile} these are pretty straight forward.")
        print("They generally include the sample ID, the previously reported phenotype and the predicted phenotype from that algorithm,")
        print("#"*70)
        print(test_out.head())
        print("#"*70)

        # Exporting training data, which is by nature overfit 
        train_predicted_values = algo.predict(self.X_train)
        train_predicted_values_df = pd.DataFrame(train_predicted_values)
        y_train_df = pd.DataFrame(self.y_train)
        IDs_train_df = pd.DataFrame(self.IDs_train)

        train_out = pd.concat([IDs_train_df.reset_index(), y_train_df.reset_index(drop=True), train_predicted_values_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        train_out.columns=["INDEX","ID","PHENO_REPORTED","PHENO_PREDICTED"]
        train_out = train_out.drop(columns=["INDEX"])

        train_outfile = self.run_prefix + '.trainedModel_trainingSample_Predictions.csv'
        train_out.to_csv(train_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the training samples which is naturally overfit and exported as {train_outfile} in the similar format as in the withheld test dataset that was just exported.")
        print("#"*70)
        print(train_out.head())
        print("#"*70)

        # Exporting regression summary
        print("")
        print("Here is a quick summary of the regression comparing PHENO_REPORTED ~ PHENO_PREDICTED in the withheld test data...")
        print("")

        reg_model = sm.ols(formula='PHENO_REPORTED ~ PHENO_PREDICTED', data=test_out)
        fitted = reg_model.fit()
        print(fitted.summary())

        print("")
        print("...always good to see the P for the predictor.")

        # Exporting regression plot
        genoML_colors = ["cyan","purple"]
        sns_plot = sns.regplot(data=test_out, y="PHENO_REPORTED", x="PHENO_PREDICTED", scatter_kws={"color": "cyan"}, line_kws={"color": "purple"})

        plot_out = self.run_prefix + '.trainedModel_withheldSample_regression.png'
        sns_plot.figure.savefig(plot_out, dpi=600)

        print("")
        print(f"We are also exporting a regression plot for you here {plot_out} this is a graphical representation of the difference between the reported and predicted phenotypes in the withheld test data for the best performing algorithm.")

    def feature_ranking(self):
        best_algo = self.best_algo
        X_train = self.X_train
        y_train = self.y_train

        if (best_algo == 'SVR') or (best_algo == 'KNeighborsRegressor'):
            print("Even if you selected to run feature ranking, you can't generate feature ranks using SVR or KNeighborsRegressor... it just isn't possible.")
        else:
            print("Processing feature ranks, this can take a while. But you will get a relative rank for every feature in the model.")

            top_ten_percent = (len(X_train)//10)
            # core_count = args.n_cores
            names = list(X_train.columns)
            rfe = RFE(estimator=self.algo)
            rfe.fit(X_train, y_train)
            rfe_out = zip(rfe.ranking_, names)
            rfe_df = pd.DataFrame(rfe_out, columns = ["RANK","FEATURE"])
            
            self.rfe_df = rfe_df
            return rfe_df

    def save_results(self, path, algorithmResults = False, bestAlgorithm = False, featureRankings = False):
        path = self.run_prefix 

        if(algorithmResults):
            log_table = self.log_table
            log_outfile = path + '.training_withheldSamples_performanceMetrics.csv'

            print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
            print("#"*70)
            print(log_table)
            print("#"*70)

            log_table.to_csv(log_outfile, index=False)

        if(bestAlgorithm):
            best_algo = self.best_algo
            print(f"Based on your withheld samples, the algorithm with the highest explained variance score is the {best_algo}... let's save that model for you.")
            best_algo_name_out = path + '.best_algorithm.txt'
            file = open(best_algo_name_out,'w')
            file.write(self.best_algo)
            file.close()
        
        if(featureRankings):
            try:
                table_outfile = path + '.trainedModel_trainingSample_featureImportance.csv'
                self.rfe_df.to_csv(table_outfile, index=False)  
                print(f"Feature ranks exported as {table_outfile} if you want to be very picky and make a more parsimonious model with a minimal feature set, extract all features ranked 1 and rebuild your dataset. This analysis also gives you a concept of the relative importance of your features in the model.")

            except:
                print("No feature rankings to save.")