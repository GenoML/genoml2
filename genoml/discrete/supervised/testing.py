class Test:
    """
    Test model's performance on the test set. 
    """
    def __init__(self, df, loaded_model, run_prefix):
        self.y_test = df.PHENO
        self.IDs_test = df.ID
        self.X_test = df.drop(columns=['PHENO', 'ID'])
        self.run_prefix = run_prefix
        self.loaded_model = loaded_model
        self.test_out = None
        self.roc_auc = None
        self.test_predictions = None
        self.test_predictions_cases = None

    def export_ROC(self):
        """
        Plot and save ROC curve
        """

        # Define the output prefix
        plot_out = self.run_prefix + '.testedModel_allSample_ROC.png'

        self.test_predictions = self.loaded_model.predict_proba(self.X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(self.y_test, self.test_predictions)
        self.roc_auc = roc_auc_score(self.y_test, self.test_predictions)

        # Plot ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='purple', label='All sample ROC curve (area = %0.2f)' % self.roc_auc + '.')
        plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Test Dataset')
        plt.legend(loc="lower right")
        plt.savefig(plot_out, dpi=600)

        print("")
        print(
            f"We are also exporting a ROC curve for you here {plot_out} this is a graphical representation of AUC in all samples for the best performing algorithm.")

    def export_tested_data(self):
        """
        Return and display table with predicted values
        """

        # Initialize predictions table
        test_case_probs_df = pd.DataFrame(self.loaded_model.predict_proba(self.X_test)[:, 1])
        test_predicted_cases_df = pd.DataFrame(self.loaded_model.predict(self.X_test))
        y_test_df = pd.DataFrame(self.y_test)
        IDs_test_df = pd.DataFrame(self.IDs_test)

        # Construct predictions table
        test_out = pd.concat([IDs_test_df.reset_index(),
                              y_test_df.reset_index(drop=True),
                              test_case_probs_df.reset_index(drop=True),
                              test_predicted_cases_df.reset_index(drop=True)],
                             axis=1, ignore_index=True
                             )
        test_out.columns = ['INDEX', 'ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
        test_out = test_out.drop(columns=['INDEX'])

        # Save table
        test_outfile = self.run_prefix + '.testedModel_allSample_predictions.csv'
        test_out.to_csv(test_outfile, index=False)

        print("")
        print(
            f"Preview of the exported predictions for the testing samples exported as {test_outfile} in the similar format as in the initial training phase of GenoML.")
        print("#" * 70)
        print(test_out.head())
        print("#" * 70)

        return test_out

    def export_histograms(self):
        """
        Save estimated distribution.
        """
        g = sns.FacetGrid(self.export_tested_data(), hue="CASE_REPORTED", palette=["cyan", "purple"], legend_out=True, )
        g = (g.map(sns.distplot, "CASE_PROBABILITY", hist=True, rug=False))
        g.add_legend()

        plot_out = self.run_prefix + '.testedModel_allSample_probabilities.png'
        g.savefig(plot_out, dpi=600)

        print("")
        print(
            f"We are also exporting probability density plots to the file {plot_out} this is a plot of the probability "
            f"distributions of being a case, stratified by case and control status for all samples.")
        print("")

    def get_metrics(self):
        """
        Return algorithm's performance metrics.
        """
        self.test_predictions_cases = self.loaded_model.predict(self.X_test)
        acc = accuracy_score(self.y_test, self.test_predictions_cases)
        balacc = balanced_accuracy_score(self.y_test, self.test_predictions_cases)
        ll = log_loss(self.y_test, self.test_predictions)

        CM = confusion_matrix(self.y_test, self.test_predictions_cases)
        TN = CM[0][0]
        FN = CM[1][0]
        TP = CM[1][1]
        FP = CM[0][1]
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        PPV = TP / (TP + FP)
        NPV = TN / (TN + FN)
        return acc, balacc, CM, ll, sensitivity, specificity, PPV, NPV

    def additional_summary_stats(self):
        """
        Construct and display algorithm performance table.
        """
        self.loaded_model.fit(self.X_test, self.y_test)

        print("")
        print("#" * 70)
        print("Some additional summary stats logging from your application of your model to the test dataset.")
        print("")

        # Get algorithm's performance metrics
        accuracy, balanced_accuracy, CM, ll, sensitivity, specificity, PPV, NPV = self.get_metrics()

        # Print Metrics
        print("AUC: {:.4%}".format(self.roc_auc))
        print("Accuracy: {:.4%}".format(accuracy))
        print("Balanced Accuracy: {:.4%}".format(balanced_accuracy))
        print("Log Loss: {:.4}".format(ll))

        # Add metrics to table
        log_cols = ["AUC_Percent", "Accuracy_Percent", "Balanced_Accuracy_Percent", "Log_Loss", "Sensitivity",
                    "Specificity", "PPV", "NPV"]
        log_table = pd.DataFrame(columns=log_cols)
        log_entry = pd.DataFrame([[self.roc_auc * 100, accuracy * 100, balanced_accuracy * 100, ll, sensitivity,
                                   specificity, PPV, NPV]],
                                 columns=log_cols)
        log_table = log_table.append(log_entry)

        print("")
        print("#" * 70)
        print("")

        log_outfile = self.run_prefix + '.testedModel_allSamples_performanceMetrics.csv'

        print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
        print("#" * 70)
        print(log_table)
        print("#" * 70)

        print("")

        #Save table
        log_table.to_csv(log_outfile, index=False)
