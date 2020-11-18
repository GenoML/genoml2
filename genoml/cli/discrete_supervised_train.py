import sys
import numpy as np
import pandas as pd

from genoml.discrete import supervised


def main(prefix, metric_max, matchingCols):
    print("")
    print("Here is some basic info on the command you are about to run.")
    print("Python Version info...")
    print(sys.version)

    # Print out chosen CLI arguments 
    print("CLI argument info...")
    print(f"Working with dataset {prefix} from previous data munging efforts.")
    print(f"You have chosen to compete the algorithms based on {metric_max}.")
    print("Give credit where credit is due, for this stage of analysis we use code from the great "
          "contributors to Python packages: argparse, xgboost, sklearn, pandas, numpy, time, matplotlib and seaborn.")
    print("As a note, in all exported probabilities and other graphics, case status is treated as a 0 or 1, with 1 "
          "representing a positive case.")
    print("")

    # Specify prefix and dataframe variables to be passed into class
    run_prefix = prefix
    infile_h5 = run_prefix + ".dataForML.h5"
    df = pd.read_hdf(infile_h5, key="dataForML")

    if matchingCols is not None:
        print(f"Looks like you are retraining your reference file. We are using "
              f"the harmonized columns you provided here: {matchingCols}")
        print(f"Note that you might have different/less features than before, given "
              f"this was harmonized between training and test dataset, and might mean your model now performs worse...")

        with open(matchingCols, 'r') as matchingCols_file:
            matching_column_names_list = matchingCols_file.read().splitlines()

        # Keep only the columns found in the file 
        df = df[np.intersect1d(df.columns, matching_column_names_list)]

    model = supervised.Train(df, run_prefix)
    model.summary()

    # Give user context prior to competing algorithms 
    # Explains to users how we are splitting their data 70:30 
    print("")
    print("Now let's compete these algorithms!")
    print("We'll update you as each algorithm runs, then summarize at the end.")
    print(
        "Here we test each algorithm under default settings using the same training and test datasets derived from a 70% training and 30% testing split of your data.")
    print("For each algorithm, we will output the following metrics...")
    print(
        "Algorithm name, hoping that's pretty self-explanatory. Plenty of resources on these common ML algorithms at https://scikit-learn.org and https://xgboost.readthedocs.io/.")
    print(
        "AUC_percent, this is the area under the curve from receiver operating characteristic analyses. This is the most common metric of classifier performance in biomedical literature, we express this as a percent. We calculate AUC based on the predicted probability of being a case.")
    print(
        "Accuracy_percent, this is the simple accuracy of the classifier, how many predictions were correct from best classification cutoff (python default).")
    print(
        "Balanced_Accuracy_Percent, consider this as the accuracy resampled to a 1:1 mix of cases and controls. Imbalanced datasets can give funny results for simple accuracy.")
    print(
        "Log_Loss, this is essentially the inverse of the likelihood function for a correct prediction, you want to minimize this.")
    print("Sensitivity, proportion of cases correctly identified.")
    print("Specificity, proportion of controls correctly identified.")
    print(
        "PPV, this is the positive predictive value, the probability that subjects with a positive result actually have the disease.")
    print(
        "NPV, this is the negative predictive value, the probability that subjects with a negative result don't have the disease.")
    print("We also log the runtimes per algorithm.")
    print("")
    print("Algorithm summaries incoming...")
    print("")

    # Compete the algorithms and construct performance table
    model.compete()

    # Choose best algorithm w.r.t. metric_max
    model.results(metric_max)

    # Fit best algorithm and save using joblib. Generate its predictions
    model.export_model()

    # Save the ROC curve image and display AUC
    model.AUC(save=True)

    # Save the estimated distribution
    model.export_prob_hist()

    # Save the algorithms' performance metrics table and best algorithm
    model.save_results(algorithmResults=True, bestAlgorithm=True)

    print("Thank you for training with GenoML!")
