# Copyright 2020 The GenoML Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sklearn
from sklearn import discriminant_analysis, ensemble, linear_model, metrics, model_selection, neighbors, neural_network, svm
import time
import xgboost


# Define the train class
class train:
    def __init__(self, df, run_prefix):
        #code that will prepare the data
        y = df.PHENO
        X = df.drop(columns=['PHENO'])
        
        # Split the data 
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, random_state=42) # 70:30
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

        #The methods we will use
        self.algorithms = [
        linear_model.LogisticRegression(solver='lbfgs'),
        ensemble.RandomForestClassifier(n_estimators=100),
        ensemble.AdaBoostClassifier(),
        ensemble.GradientBoostingClassifier(),
        linear_model.SGDClassifier(loss='modified_huber'),
        svm.SVC(probability=True, gamma='scale'),
        neural_network.MLPClassifier(),
        neighbors.KNeighborsClassifier(),
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        ensemble.BaggingClassifier(),
        xgboost.XGBClassifier()
        ]
    
    # Report and data summary you want 
    def summary(self):
        print("Your data looks like this (showing the first few lines of the left-most and right-most columns)...")
        print("#"*70)
        print(self.df.describe())
        print("#"*70)

    def compete(self, verbose=False):
        log_cols=["Algorithm", "AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV", "Runtime_Seconds"]
        log_table = pd.DataFrame(columns=log_cols)

        for algo in self.algorithms:
            
            start_time = time.time()
            
            algo.fit(self.X_train, self.y_train)
            name = algo.__class__.__name__

            print("")
            print("#"*70)
            print("")
            print(name)

            test_predictions = algo.predict_proba(self.X_test)
            self.test_predictions = test_predictions
            test_predictions = test_predictions[:, 1]
            rocauc = metrics.roc_auc_score(self.y_test, test_predictions)
            print("AUC: {:.4%}".format(rocauc))

            test_predictions = algo.predict(self.X_test)
            acc = metrics.accuracy_score(self.y_test, test_predictions)
            print("Accuracy: {:.4%}".format(acc))

            test_predictions = algo.predict(self.X_test)
            balacc = metrics.balanced_accuracy_score(self.y_test, test_predictions)
            print("Balanced Accuracy: {:.4%}".format(balacc))
            
            CM = metrics.confusion_matrix(self.y_test, test_predictions)
            TN = CM[0][0]
            FN = CM[1][0]
            TP = CM[1][1]
            FP = CM[0][1]
            sensitivity = TP/(TP+FN)
            specificity = TN/(TN+FP)
            PPV = TP/(TP+FP)
            NPV = TN/(TN+FN)
            
            test_predictions = self.test_predictions
            #test_predictions = algo.predict_proba(self.X_test)
            #self.test_predictions = test_predictions
            ll = metrics.log_loss(self.y_test, test_predictions)
            print("Log Loss: {:.4}".format(ll))
            
            end_time = time.time()
            elapsed_time = (end_time - start_time)
            print("Runtime in seconds: {:.4}".format(elapsed_time))

            log_entry = pd.DataFrame([[name, rocauc*100, acc*100, balacc*100, ll, sensitivity, specificity, PPV, NPV, elapsed_time]], columns=log_cols)
            log_table = log_table.append(log_entry)

        print("#"*70)
        print("")

        self.log_table = log_table

        return log_table

    def results(self, metric_max):
        self.metric_max = metric_max 

        if(metric_max == "AUC"):
            best_performing_summary = self.log_table[self.log_table.AUC_Percent == self.log_table.AUC_Percent.max()]
            best_algo = best_performing_summary.at[0,'Algorithm']
        
        if(metric_max == "Balanced_Accuracy"):
            best_performing_summary = self.log_table[self.log_table.Balanced_Accuracy_Percent == self.log_table.Balanced_Accuracy_Percent.max()]
            best_algo = best_performing_summary.at[0,'Algorithm']
        
        if(metric_max == "Sensitivity"):
            best_performing_summary = self.log_table[self.log_table.Sensitivity == self.log_table.Sensitivity.max()]
            best_algo = best_performing_summary.at[0,'Algorithm']           
        
        if(metric_max == "Specificity"):
            best_performing_summary = self.log_table[self.log_table.Specificity == self.log_table.Specificity.max()]
            best_algo = best_performing_summary.at[0,'Algorithm']
        
        # If, for some reason, algorithms report the exact same score, only choose the first one so things don't crash
        if isinstance(best_algo, list):
            best_algo = best_algo[0]
        
        self.best_algo = best_algo

        return best_algo


    def AUC(self, save = False):
        plot_out = self.run_prefix + '.trainedModel_withheldSample_ROC.png'

        # Issue #24: RandomForestClassifier is finicky - can't recalculate moving forward like the other 
        if(self.best_algo == 'RandomForestClassifier'):
            test_predictions = self.test_predictions
            self.test_predictions = test_predictions
            test_predictions = test_predictions[:, 1]
        else:
            test_predictions = self.algo.predict_proba(self.X_test)
            self.test_predictions = test_predictions
            test_predictions = test_predictions[:, 1]

        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, test_predictions)
        #roc_auc = metrics.auc(fpr, tpr)
        roc_auc = metrics.roc_auc_score(self.y_test, test_predictions)

        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('Receiver operating characteristic (ROC) - ' + self.best_algo)
        plt.legend(loc="lower right")
        if (save):
            plt.savefig(plot_out, dpi = 600)

        #print()
        print(f"We are also exporting a ROC curve for you here {plot_out} this is a graphical representation of AUC in the withheld test data for the best performing algorithm.")
    
    def export_prob_hist(self):
        # Exporting withheld test data
        #test_predicteds_probs = self.algo.predict_proba(self.X_test)
        test_predicteds_probs = self.test_predictions
        test_case_probs = test_predicteds_probs[:, 1]
        test_predicted_cases = self.algo.predict(self.X_test)

        test_case_probs_df = pd.DataFrame(test_case_probs)
        test_predicted_cases_df = pd.DataFrame(test_predicted_cases)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        test_out = pd.concat([IDs_test_df.reset_index(), y_test_df.reset_index(drop=True), test_case_probs_df.reset_index(drop=True), test_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        test_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        test_out = test_out.drop(columns=['INDEX'])

        test_outfile = self.run_prefix + '.trainedModel_withheldSample_Predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the withheld test data that has been exported as {test_outfile} these are pretty straight forward.")
        print("They generally include the sample ID, the previously reported case status (1 = case), the case probability from the best performing algorithm and the predicted label from that algorithm")
        print("")
        print("#"*70)
        print(test_out.head())
        print("#"*70)


        # Exporting training data, which is by nature overfit
        train_predicteds_probs = self.algo.predict_proba(self.X_train)
        train_case_probs = train_predicteds_probs[:, 1]
        train_predicted_cases = self.algo.predict(self.X_train)

        train_case_probs_df = pd.DataFrame(train_case_probs)
        train_predicted_cases_df = pd.DataFrame(train_predicted_cases)
        y_train_df = pd.DataFrame(self.y_train)
        IDs_train_df = pd.DataFrame(self.IDs_train)

        train_out = pd.concat([IDs_train_df.reset_index(), y_train_df.reset_index(drop=True), train_case_probs_df.reset_index(drop=True), train_predicted_cases_df.reset_index(drop=True)], axis = 1, ignore_index=True)
        train_out.columns=['INDEX','ID',"CASE_REPORTED","CASE_PROBABILITY","CASE_PREDICTED"]
        train_out = train_out.drop(columns=['INDEX'])

        train_outfile = self.run_prefix + '.trainedModel_trainingSample_Predictions.csv'
        train_out.to_csv(train_outfile, index=False)

        print("")
        print(f"Preview of the exported predictions for the training samples which is naturally overfit and exported as {train_outfile} in the similar format as in the withheld test dataset that was just exported.")
        print("#"*70)
        print(train_out.head())
        print("#"*70)

        # Export historgrams of probabilities
        genoML_colors = ["cyan","purple"]

        g = sns.FacetGrid(train_out, hue="CASE_REPORTED", palette=genoML_colors, legend_out=True,)
        g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=True, rug=False))
        g.add_legend()

        plot_out = self.run_prefix + '.trainedModel_withheldSample_probabilities.png'
        g.savefig(plot_out, dpi=600)

        print("")
        print("We are also exporting probability density plots to the file", plot_out, "this is a plot of the probability distributions of being a case, stratified by case and control status in the withheld test samples.")

    def export_model(self):
        best_algo = self.best_algo

        if best_algo == 'LogisticRegression':
            algo = getattr(sklearn.linear_model, best_algo)()

        elif best_algo == 'SGDClassifier':
            algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

        elif (best_algo == 'RandomForestClassifier') or (best_algo == 'AdaBoostClassifier') or (best_algo == 'GradientBoostingClassifier') or  (best_algo == 'BaggingClassifier'):
            algo = getattr(sklearn.ensemble, best_algo)()

        elif best_algo == 'SVC':
            algo = getattr(sklearn.svm, best_algo)(probability=True)

        elif best_algo == 'ComplementNB':
            algo = getattr(sklearn.naive_bayes, best_algo)()

        elif best_algo == 'MLPClassifier':
            algo = getattr(sklearn.neural_network, best_algo)()

        elif best_algo == 'XGBClassifier':
            algo = getattr(xgboost, best_algo)()

        elif best_algo == 'KNeighborsClassifier':
            algo = getattr(sklearn.neighbors, best_algo)()

        elif (best_algo == 'LinearDiscriminantAnalysis') or (best_algo == 'QuadraticDiscriminantAnalysis'):
            algo = getattr(sklearn.discriminant_analysis, best_algo)()

        algo.fit(self.X_train, self.y_train)
        name = algo.__class__.__name__

        print("...remember, there are occasionally slight fluctuations in model performance on the same withheld samples...")
        print("#"*70)
        print(name)

        #test_predictions = algo.predict_proba(self.X_test)
        test_predictions = self.test_predictions
        test_predictions = test_predictions[:, 1]
        rocauc = metrics.roc_auc_score(self.y_test, test_predictions)
        print("AUC: {:.4%}".format(rocauc))

        test_predictions = algo.predict(self.X_test)
        acc = metrics.accuracy_score(self.y_test, test_predictions)
        print("Accuracy: {:.4%}".format(acc))

        test_predictions = algo.predict(self.X_test)
        balacc = metrics.balanced_accuracy_score(self.y_test, test_predictions)
        print("Balanced Accuracy: {:.4%}".format(balacc))

        #test_predictions = algo.predict_proba(self.X_test)
        test_predictions = self.test_predictions
        ll = metrics.log_loss(self.y_test, test_predictions)
        print("Log Loss: {:.4}".format(ll))

        ### Save it using joblib
        algo_out = self.run_prefix + '.trainedModel.joblib'
        joblib.dump(algo, algo_out)

        print("#"*70)
        print(f"... this model has been saved as {algo_out} for later use and can be found in your working directory.")

        self.algo = algo

        return algo

    def save_results(self, run_prefix, algorithmResults = False, bestAlgorithm = False, featureRankings = False):
        if(algorithmResults):
            log_table = self.log_table
            log_outfile = self.run_prefix + '.training_withheldSamples_performanceMetrics.csv'

            print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
            print("#"*70)
            print(log_table)
            print("#"*70)

            log_table.to_csv(log_outfile, index=False)

        if(bestAlgorithm):
            best_algo = self.best_algo
            print(f"Based on your withheld samples, the algorithm with the best {self.metric_max} is the {best_algo}... let's save that model for you.")
            best_algo_name_out = self.run_prefix + ".best_algorithm.txt"
            file = open(best_algo_name_out,'w')
            file.write(self.best_algo)
            file.close() 
