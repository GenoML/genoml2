import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model, metrics, model_selection
import xgboost
from scipy import stats
import seaborn as sns

class Classification:
    """
    This class trains, tunes, and tests a logistic regression and XGBoost classifier on a given dataset.
    """
    def __init__(
            self,
            df: pd.DataFrame()):
        # Features matrix
        X = df.drop(columns=['PHENO'])
        # PHENO Vector
        y = df.PHENO
        # Split the data: 80% to train set, 20% to test set
        X_train, X_test, y_train, self.y_test = model_selection.train_test_split(X, y,
                                                                                      test_size=0.2,
                                                                                      random_state=42)
        self.IDs_test = X_test.ID
        # Split the data: 60% to train set, 20% to validation set
        X_train, X_val, self.y_train, self.y_val= model_selection.train_test_split(X_train, y_train,
                                                                                             test_size=0.25,
                                                                                             random_state=42)
        self.X_train = X_train.drop(columns=['ID'])
        self.X_test = X_test.drop(columns=['ID'])
        self.X_val = X_val.drop(columns=['ID'])
        self.log_reg = linear_model.LogisticRegression(solver='lbfgs')
        self.xgb = xgboost.XGBClassifier()
        self.hyperparameters = dict({'LogisticRegression': {"penalty": ['l1', 'l2'], "C": stats.randint(1, 10)},
                                     'XGBClassifier': {"max_depth": stats.randint(1, 100),
                                                       "learning_rate": stats.uniform(0, 1),
                                                       "n_estimators": stats.randint(1, 100),
                                                       "gamma": stats.uniform(0, 1)}})


    def fit(self, algorithm):
        """
        Fits given algorithm to the training set.
        """
        return algorithm.fit(self.X_train, self.y_train)

    def tune(self, fitted_algorithm, max_iter=50, cv_count=5):
        """
        Tunes Hyperparameters for given algorithm with respect to performance metric (AUC) on validation set.
        Iterates until convergence a maximum number of max_iter times.
        """
        name = fitted_algorithm.__class__.__name__
        scoring_metric = metrics.make_scorer(metrics.roc_auc_score, needs_proba=True)
        random_search = model_selection.RandomizedSearchCV(estimator=fitted_algorithm,
                                                              param_distributions=self.hyperparameters[name],
                                                              scoring=scoring_metric, n_iter=max_iter,
                                                              cv=cv_count, n_jobs=-1, random_state=153, verbose=0)
        random_search.fit(self.X_val, self.y_val)
        return random_search.best_estimator_

    def test(self, tuned_algorithm):
        """
        Returns predictions of the tuned algorithm on the test set.
        """
        predicted_probabilities = tuned_algorithm.predict_proba(self.X_test)
        predicted_classes = tuned_algorithm.predict(self.X_test)
        return predicted_probabilities, predicted_classes

    def metrics_table(self, predicted_probabilities, predicted_classes):
        """
        Returns data frame with the algorithm's performance metrics.
        """
        results = self.computeMetrics(predicted_probabilities, predicted_classes)
        return pd.DataFrame.from_dict(results, orient='index')

    def computeMetrics(self, predicted_probabilities, predicted_classes):
        """
        This function calculates the relevant metrics to test
        the classification algorithms' performances.
        """
        roc_auc = metrics.roc_auc_score(self.y_test, predicted_probabilities[:, 1])
        accuracy = metrics.accuracy_score(self.y_test, predicted_classes)
        balanced_accuracy_score = metrics.balanced_accuracy_score(self.y_test, predicted_classes)
        ll = metrics.log_loss(self.y_test, predicted_probabilities)
        CM = metrics.confusion_matrix(self.y_test, predicted_classes)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        metrics_names =  ["AUC_Percent", "Accuracy_Percent",
                         "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity",
                         "Specificity", "PPV", "NPV"]
        metrics_results = [roc_auc, accuracy, balanced_accuracy_score, ll, sensitivity, specificity, PPV, NPV]
        return dict(zip(metrics_names, metrics_results))

    def ROC_curve(self, predicted_probabilities, path):
        """
        Plot and save ROC curve.
        """
        fpr, tpr, thresholds = metrics.roc_curve(self.y_test, predicted_probabilities[:,1])
        roc_auc = metrics.roc_auc_score(self.y_test, predicted_probabilities[:,1])
        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % roc_auc + '.')
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Test Dataset')
        plt.legend(loc="lower right")
        # Save ROC curve
        plt.savefig(path + 'ROC.png', dpi=600)

    def tested_data(self, predicted_probabilities, predicted_classes):
        """
        return table with predicted values
        """
        # Initialize predictions table
        predicted_probabilities_df = pd.DataFrame(predicted_probabilities[:, 1])
        predicted_classes_df = pd.DataFrame(predicted_classes)
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)
        # Construct predictions table
        test_out = pd.concat([IDs_test_df.reset_index(),
                              y_test_df.reset_index(drop=True),
                              predicted_probabilities_df.reset_index(drop=True),
                              predicted_classes_df.reset_index(drop=True)],
                             axis=1, ignore_index=True
                             )
        test_out.columns = ['INDEX', 'ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
        return test_out.drop(columns=['INDEX'])

    def histogram(self, predicted_probabilities, predicted_classes, path):
        """
        Save estimated distribution.
        """
        tested_data = self.tested_data(predicted_probabilities, predicted_classes)
        histogram = sns.FacetGrid(tested_data, hue="CASE_REPORTED", palette=["cyan", "purple"], legend_out=True)
        histogram = (histogram.map(sns.distplot, "CASE_PROBABILITY", hist=True, rug=False))
        histogram.add_legend()
        #Save graph on given path
        plot_out = path + 'histogram.png'
        histogram.savefig(plot_out, dpi=600)