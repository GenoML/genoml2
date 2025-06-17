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

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from genoml import utils


### TODO: Inputs should be numpy instead of pandas?
def plot_results(out_dir, y, y_pred, algorithm_name):
    """
    Generate ROC and precision-recall plots for each class.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted phenotype probabilities.
        algorithm_name (str): Name of the algorithm used for prediction.

    :return: num_classes *(int)*: \n
        Number of classes being predicted
    """

    roc_path = out_dir.joinpath('roc.png')
    precision_recall_path = out_dir.joinpath('precision_recall.png')
    num_classes = y_pred.shape[1]
    ROC(roc_path, y.values, y_pred, algorithm_name, num_classes)
    precision_recall_plot(precision_recall_path, y.values, y_pred, algorithm_name, num_classes)
    return num_classes


def ROC(plot_path, y, y_pred, algorithm_name, num_classes):
    """
    Generate ROC plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred (numpy.ndarray): Predicted probabilities for each class.
        algorithm_name (str): Label to add to plot title.
        num_classes (int): Number of classes being predicted.
    """

    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')

    for i in range(num_classes):
        fpr, tpr, _ = metrics.roc_curve(y[:, i], y_pred[:, i])
        roc_auc = metrics.roc_auc_score(y[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f"Class {i + 1} (area = {roc_auc})")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Receiver operating characteristic (ROC) - {algorithm_name}')
    plt.legend(loc="lower right")
    plt.savefig(plot_path, dpi=600)
    print(f"We are also exporting an ROC curve for you here {plot_path} this is a graphical representation of AUC "
          f"in the withheld test data for the best performing algorithm.")


def precision_recall_plot(plot_path, y, y_pred, algorithm_name, num_classes):
    """
    Generate precision-recall plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (str): File path where plot will be saved to.
        y (numpy.ndarray): Ground truth phenotypes.
        y_pred (numpy.ndarray): Predicted probabilities for each class.
        algorithm_name (str): Label to add to plot title.
        num_classes (int): Number of classes being predicted.
    """

    plt.figure()

    for i in range(num_classes):
        precision, recall, _ = metrics.precision_recall_curve(y[:, i], y_pred[:, i])
        plt.plot(precision, recall, label=f"Class {i + 1}")

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


def export_prediction_data(out_dir, y, y_pred, ids, num_classes, y_train=None, y_train_pred=None, ids_train=None):
    """
    Save probability histograms and tables with accuracy metrics

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted probabilities for each class.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        num_classes (int): Number of classes being predicted.
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
            num_classes,
            dataset="training",
        )

    df_prediction = export_prediction_tables(
        y,
        y_pred,
        ids,
        out_dir.joinpath('predictions.txt'),
        num_classes,
    )

    export_prob_hist(
        num_classes,
        df_prediction,
        out_dir.joinpath('probabilities'),
    )


def calculate_accuracy_scores(x, y_proba, algorithm):
    """
    Calculate accuracy metrics for the chosen multiclass prediction model.

    Args:
        x (pandas.DataFrame): Model input features.
        y_proba (pandas.DataFrame): Reported output features.
        algorithm: Contonuous prediction algorithm.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the multiclass prediction module.
    """

    y_pred_proba = algorithm.predict_proba(x)
    return _calculate_accuracy_scores(y_proba, y_pred_proba)


### TODO: Macro vs weighted? Separate for each class using one vs all?
def _calculate_accuracy_scores(y_proba, y_pred_proba):
    """
    Calculate accuracy metrics for the chosen multiclass prediction model.

    Args:
        y_proba (pandas.DataFrame): Reported output features.
        y_pred_proba (pandas.DataFrame): Predicted output features.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the multiclass prediction module.
    """

    y_argmax = y_proba.values.argmax(axis=1)
    y_pred_argmax = y_pred_proba.argmax(axis=1)

    rocauc = metrics.roc_auc_score(y_proba, y_pred_proba, multi_class="ovr") * 100
    acc = metrics.accuracy_score(y_argmax, y_pred_argmax) * 100
    balacc = metrics.balanced_accuracy_score(y_argmax, y_pred_argmax) * 100
    ll = metrics.log_loss(y_proba, y_pred_proba)
    
    n_classes = y_proba.shape[1]
    sens = spec = ppv = npv = 0
    for class_ in range(n_classes):
        y_vals_class = np.where(y_argmax == class_, 1, 0)
        y_vals_pred_class = np.where(y_pred_argmax == class_, 1, 0)
        CM = metrics.confusion_matrix(y_vals_class, y_vals_pred_class)
        tn = CM[0][0]
        fn = CM[1][0]
        tp = CM[1][1]
        fp = CM[0][1]
        sens += tp / (tp+fn) / n_classes
        spec += tn / (tn+fp) / n_classes
        ppv += tp / (tp+fp) / n_classes
        npv += tn / (tn+fn) / n_classes

    accuracy_metrics = [rocauc, acc, balacc, ll, sens, spec, ppv, npv]
    return accuracy_metrics


def export_prediction_tables(y, y_pred, ids, output_path, num_classes, dataset="withheld test"):
    """
    Generate and save tables with prediction probabilities and predicted classes for each sample.

    Args:
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred (pandas.DataFrame): Predicted phenotypes.
        ids (pandas.Series): ids for participants corresponding to the datasets.
        output_path (pathlib.Path): Where to save output files.
        num_classes (int): Number of classes being predicted.
        dataset (str): Indicator of whether analyzing training, tuning, or testing data.

    :return: df_prediction *(pandas.DataFrame)*: \n
        Table of reported and predicted phenotypes.
    """

    y_pred = pd.DataFrame(y_pred)
    df_predicted_cases = y_pred.idxmax(axis=1)
    y = pd.DataFrame(y).idxmax(axis=1)
    ids = pd.DataFrame(ids)
    df_prediction = pd.concat(
        [
            ids.reset_index(drop=True),
            y.reset_index(drop=True),
            y_pred.reset_index(drop=True),
            df_predicted_cases.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )

    df_prediction.columns = ["ID", "CASE_REPORTED"] \
                            + [f"CASE{i + 1}_PROBABILITY" for i in range(num_classes)] \
                            + ["CASE_PREDICTED"]
    df_prediction.to_csv(output_path, index=False, sep="\t")

    print("")
    print(f"Preview of the exported predictions for the {dataset} data that has been exported as {output_path}.")
    print("")
    print("#" * 70)
    print(df_prediction.head())
    print("#" * 70)

    return df_prediction


def export_prob_hist(num_classes, df_plot, plot_prefix):
    """
    Save probability histograms for each class.

    Args:
        num_classes (int): Number of classes being predicted.
        df_plot (pandas.DataFrame): Table of predicted phenotypes.
        plot_prefix (pathlib.Path): Prefix for output files.
    """

    for i in range(num_classes):
        df_plot[f'Probability{i + 1} (%)'] = (df_plot[f'CASE{i + 1}_PROBABILITY'] * 100).round(decimals=0)
    df_plot['Reported Status'] = df_plot['CASE_REPORTED']
    df_plot['Predicted Status'] = df_plot['CASE_PREDICTED']

    # Start plotting
    plt.figure()
    colors = sns.husl_palette(num_classes)
    for i in range(num_classes):
        sns.histplot(
            data = df_plot,
            x = f"Probability{i + 1} (%)",
            hue = "Predicted Status",
            kde = True,
            palette = colors,
            alpha = 0.2,
            multiple = 'dodge',
        )
        path = f"{plot_prefix}{i+1}.png"
        plt.savefig(path, dpi=300)
        plt.clf()
        print(f"We are also exporting probability density plots to the file {path} this is a plot of the probability "
              f"distributions for each case, stratified by case status in the withheld test samples.")
