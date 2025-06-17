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

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import sklearn
from sklearn import metrics
import seaborn as sns
import time
import xgboost


### TODO: Inputs should be numpy instead of pandas?
def plot_results(out_dir, y, y_pred, algorithm_name):
    """
    Generate ROC and precision-recall plots for each class.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted phenotypes.
        algorithm_name: Classifier model in OneVsRestClassifier wrapper.
    """

    y_pred = y_pred.argmax(axis=1)
    roc_path = out_dir.joinpath('roc.png')
    precision_recall_path = out_dir.joinpath('precision_recall.png')
    ROC(roc_path, y.values, y_pred, algorithm_name)
    precision_recall_plot(precision_recall_path, y.values, y_pred, algorithm_name)


def ROC(plot_path, y, y_pred, algorithm_name):
    """
    Generate ROC plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred (numpy.ndarray): Predicted probabilities for each class.
        algorithm_name (str): Label to add to plot title.
    """

    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')

    fpr, tpr, _ = metrics.roc_curve(y, y_pred)
    roc_auc = metrics.roc_auc_score(y, y_pred)
    plt.plot(fpr, tpr, color='purple', label=f'ROC curve (area = {roc_auc})')

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Receiver operating characteristic (ROC) - {algorithm_name}')
    plt.legend(loc="lower right")
    plt.savefig(plot_path, dpi=600)
    print(f"We are also exporting an ROC curve for you here {plot_path} this is a graphical representation of AUC "
          f"in the withheld test data for the best performing algorithm.")


def precision_recall_plot(plot_path, y, y_pred, algorithm_name):
    """
    Generate precision-recall plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred (numpy.ndarray): Predicted probabilities for each class.
        algorithm_name (str): Label to add to plot title.
    """

    plt.figure()

    precision, recall, _ = metrics.precision_recall_curve(y, y_pred)
    plt.plot(precision, recall, label="Precision-Recall curve")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Recall Curve - {algorithm_name}")
    plt.legend(loc="lower left")
    plt.savefig(plot_path, dpi=600)
    print(f"We are also exporting a Precision-Recall plot for you here {plot_path}. This is a graphical "
          f"representation of the relationship between precision and recall scores in the withheld test data for "
          f"the best performing algorithm.")


def export_prediction_data(out_dir, y, y_pred, ids, y_train=None, y_train_pred=None, ids_train=None):
    """
    Save probability histograms and tables with accuracy metrics.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted probabilities for each class.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        y_train (optional, pandas.DataFrame): Ground truth phenotypes from the training dataset (Default: None).
        y_train_pred (optional, pandas.DataFrame): Predicted phenotypes from the training dataset (Default: None).
        ids_train (optional, pandas.Series): ids for participants in the training dataset (Default: None).
    """

    if y_train is not None and y_train_pred is not None and ids_train is not None:
        export_prediction_tables(
            y_train,
            y_train_pred,
            ids_train,
            out_dir.joinpath('train_predictions.txt'),
            dataset="training",
        )

    df_prediction = export_prediction_tables(
        y,
        y_pred,
        ids,
        out_dir.joinpath('predictions.txt'),
    )

    export_prob_hist(
        df_prediction,
        out_dir.joinpath('probabilities'),
    )


def calculate_accuracy_scores(x, y, algorithm):
    """
    Apply discrete prediction model and calculate accuracy metrics.

    Args:
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        algorithm: Contonuous prediction algorithm.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the discrete prediction module.
    """

    y_pred = algorithm.predict(x)
    return _calculate_accuracy_scores(y, y_pred)


def _calculate_accuracy_scores(y, y_pred):
    """
    Calculate accuracy metrics for the chosen discrete prediction model.

    Args:
        y (pandas.DataFrame): Reported output features.
        y_pred (pandas.DataFrame): Predicted output features.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the discrete prediction module.
    """

    rocauc = metrics.roc_auc_score(y, y_pred) * 100
    acc = metrics.accuracy_score(y, y_pred) * 100
    balacc = metrics.balanced_accuracy_score(y, y_pred) * 100
    ll = metrics.log_loss(y, y_pred)
    
    CM = metrics.confusion_matrix(y, y_pred)
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    sens = TP/(TP+FN)
    spec = TN/(TN+FP)
    ppv = TP/(TP+FP)
    npv = TN/(TN+FN)

    accuracy_metrics = [rocauc, acc, balacc, ll, sens, spec, ppv, npv]
    return accuracy_metrics


def export_prediction_tables(y, y_pred, ids, output_path, dataset="withheld test"):
    """
    Generate and save tables with prediction probabilities and predicted classes for each sample.

    Args:
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted phenotypes.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        output_path (pathlib.Path): Where to save output files.
        dataset (str): Indicator of whether analyzing training, tuning, or testing data.

    :return: df_prediction *(pandas.DataFrame)*: \n
        Table of reported and predicted phenotypes.
    """

    y_pred = pd.DataFrame(y_pred, dtype=float)
    df_predicted_cases = y_pred.idxmax(axis=1)
    case_probs = pd.DataFrame(y_pred.iloc[:,1])
    y = pd.DataFrame(y)
    ids = pd.DataFrame(ids)
    df_prediction = pd.concat(
        [
            ids.reset_index(drop=True),
            y.reset_index(drop=True),
            case_probs.reset_index(drop=True),
            df_predicted_cases.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )

    df_prediction.columns = ['ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
    df_prediction.to_csv(output_path, index=False, sep="\t")

    print("")
    print(f"Preview of the exported predictions for the {dataset} data that has been exported as {output_path}.")
    print("")
    print("#" * 70)
    print(df_prediction.head())
    print("#" * 70)

    return df_prediction


def export_prob_hist(df_plot, plot_prefix):
    """
    Save probability histograms for each class.

    Args:
        df_plot (pandas.DataFrame): Table of predicted phenotypes.
        plot_prefix (pathlib.Path): Prefix for output files.
    """

    # Using the withheld sample data
    df_plot[f'Probability (%)'] = (df_plot[f'CASE_PROBABILITY'] * 100).round(decimals=0)
    df_plot['Reported Status'] = df_plot['CASE_REPORTED']
    df_plot['Predicted Status'] = df_plot['CASE_PREDICTED']

    # Start plotting
    plt.figure()
    sns.histplot(
        data=df_plot,
        x=f"Probability (%)",
        hue="Predicted Status",
        kde=True,
        alpha=0.2,
        multiple='dodge',
    )
    path = f"{plot_prefix}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(f"We are also exporting probability density plots to the file {path} this is a plot of the probability "
          f"distributions for each case, stratified by case status in the withheld test samples.")
