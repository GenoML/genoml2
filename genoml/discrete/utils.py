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
import numpy as np
import pandas as pd
import seaborn as sns
from genoml import utils
from sklearn import metrics


def plot_results(out_dir, y, x, algorithm, is_using_outer_cv, is_testing=False):
    """
    Generate ROC and precision-recall plots for each class.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        y (numpy.ndarray | list): Ground truth phenotypes (list of arrays when using outer CV).
        x (numpy.ndarray | list): Input values (list of arrays when using outer CV).
        algorithm: Discrete prediction algorithm (or list of algorithms when using outer CV).
        is_using_outer_cv (bool): Whether outer cross-validation is active.
        is_testing (bool): Whether we are in testing mode.
    """
    if is_using_outer_cv:
        algorithm_name = utils.get_algorithm_name(algorithm[0])

        if is_testing:
            # All folds combined into one pair of plots.
            fold_data = [
                (y[fold], algo.predict_proba(x[fold])[:, 1], f"Fold {fold + 1}")
                for fold, algo in enumerate(algorithm)
            ]
            ROC(out_dir.joinpath("roc.png"), fold_data, algorithm_name)
            precision_recall_plot(out_dir.joinpath("precision_recall.png"), fold_data, algorithm_name)

        else:
            # Fold 0 is the full-data fit — plot it on its own.
            full_pred_prob = algorithm[0].predict_proba(x[0])[:, 1]
            ROC(
                out_dir.joinpath("roc.png"),
                [(y[0], full_pred_prob, None)],
                algorithm_name,
            )
            precision_recall_plot(
                out_dir.joinpath("precision_recall.png"),
                [(y[0], full_pred_prob, None)],
                algorithm_name,
            )

            # Folds 1-N combined into the CV pair of plots.
            cv_data = [
                (y[fold], algorithm[fold].predict_proba(x[fold])[:, 1], f"Fold {fold}")
                for fold in range(1, len(algorithm))
            ]
            ROC(out_dir.joinpath("roc_cv.png"), cv_data, algorithm_name)
            precision_recall_plot(out_dir.joinpath("precision_recall_cv.png"), cv_data, algorithm_name)

    else:
        algorithm_name = utils.get_algorithm_name(algorithm)
        y_pred_prob = algorithm.predict_proba(x)[:, 1]
        ROC(out_dir.joinpath("roc.png"), [(y, y_pred_prob, None)], algorithm_name)
        precision_recall_plot(out_dir.joinpath("precision_recall.png"), [(y, y_pred_prob, None)], algorithm_name)


def ROC(plot_path, fold_data, algorithm_name):
    """
    Generate ROC plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (pathlib.Path): File path where the plot will be saved.
        fold_data (list[tuple]): List of (y, y_pred_prob, label) tuples. Pass label=None for single-curve (non-CV) plots.
        algorithm_name (str): Label added to the plot title.
    """
    plot_rows = []

    for y, y_pred_prob, label in fold_data:
        fpr, tpr, _ = metrics.roc_curve(y, y_pred_prob)
        roc_auc = metrics.roc_auc_score(y, y_pred_prob)
        full_label = f"{label} (AUC = {roc_auc:.3f})" if label else f"ROC curve (AUC = {roc_auc:.3f})"
        fold_df = pd.DataFrame({
            "FPR": fpr,
            "TPR": tpr,
            "Curve": full_label
        })
        plot_rows.append(fold_df)

    df_roc = pd.concat(plot_rows, ignore_index=True)

    # 2. Generate the plot
    plt.figure(figsize=(8, 6))
    sns.set_theme(style="whitegrid")
    plt.plot([0, 1], [0, 1], color='darkgrey', linestyle='--', label='_nolegend_')
    sns.lineplot(
        data=df_roc,
        x="FPR",
        y="TPR",
        hue="Curve",
        palette="tab10",
        linewidth=1.5,
        drawstyle="steps-post",
        errorbar=None,
        estimator=None,
    )
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Receiver operating characteristic (ROC) - {algorithm_name}', fontsize=10)
    plt.legend(loc="lower right")

    plt.savefig(plot_path, dpi=600)
    print(
        f"We are also exporting an ROC curve for you here {plot_path} this is a graphical representation of AUC "
        f"in the withheld test data for the best performing algorithm."
    )


def precision_recall_plot(plot_path, fold_data, algorithm_name):
    """
    Generate precision-recall plots for each class given ground-truth values and corresponding predictions.

    Args:
        plot_path (pathlib.Path): File path where the plot will be saved.
        fold_data (list[tuple]): List of (y, y_pred_prob, label) tuples. Pass label=None for single-curve (non-CV) plots.
        algorithm_name (str): Label added to the plot title.
    """
    plt.figure()
    fold_colors = sns.color_palette("tab10", n_colors=len(fold_data))

    for i, (y, y_pred_prob, label) in enumerate(fold_data):
        precision, recall, _ = metrics.precision_recall_curve(y, y_pred_prob)
        line_label = label if label else "Precision-Recall curve"
        plt.plot(recall, precision, color=fold_colors[i], label=line_label)

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Recall Curve - {algorithm_name}", fontsize=10)
    plt.legend(loc="lower left")
    plt.savefig(plot_path, dpi=600)
    print(
        f"We are also exporting a Precision-Recall plot for you here {plot_path}. This is a graphical "
        f"representation of the relationship between precision and recall scores in the withheld test data for "
        f"the best performing algorithm."
    )


def export_prediction_data(out_dir, algorithm, y, x, ids, is_using_outer_cv, y_train=None, x_train=None, ids_train=None, is_testing=False):
    """
    Save probability histograms and tables with accuracy metrics.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        algorithm: Discrete prediction algorithm (or list when using outer CV).
        y (pandas.DataFrame | list): Ground truth phenotypes.
        x (pandas.DataFrame | list): Input data.
        ids (pandas.Series | list): Participant IDs.
        is_using_outer_cv (bool): Whether outer cross-validation is active.
        y_train (optional, pandas.DataFrame | list): Training ground truth (Default: None).
        x_train (optional, pandas.DataFrame | list): Training input data (Default: None).
        ids_train (optional, pandas.Series | list): Training participant IDs (Default: None).
        is_testing (bool): Whether we are in testing mode.
    """

    if is_using_outer_cv:
        if is_testing:
            # All folds combined into one predictions file.
            combined_dfs = []
            for fold, algo in enumerate(algorithm):
                fold_num = fold + 1
                y_pred_prob = algo.predict_proba(x[fold])
                df = _build_prediction_df(y[fold], y_pred_prob, ids[fold])
                df["FOLD"] = fold_num
                combined_dfs.append(df)
                export_prob_hist(df.copy(), out_dir.joinpath(f"probabilities_fold{fold_num}"))
                if y_train is not None and x_train is not None and ids_train is not None:
                    export_prediction_tables(
                        y_train[fold],
                        algo.predict_proba(x_train[fold]),
                        ids_train[fold],
                        out_dir.joinpath(f"train_predictions_fold{fold_num}.tsv"),
                        dataset="training",
                    )
            _save_and_preview_predictions(
                pd.concat(combined_dfs, ignore_index=True),
                out_dir.joinpath("predictions.tsv"),
            )

        else:
            # Fold 0 is the full-data fit — write it on its own.
            _export_prediction_data(
                out_dir, algorithm[0], y[0], x[0], ids[0],
                y_train=y_train[0] if y_train is not None else None,
                x_train=x_train[0] if x_train is not None else None,
                ids_train=ids_train[0] if ids_train is not None else None,
                fold=0,
            )

            # Folds 1-N combined into the CV predictions file.
            combined_dfs = []
            for fold in range(1, len(algorithm)):
                algo = algorithm[fold]
                y_pred_prob = algo.predict_proba(x[fold])
                df = _build_prediction_df(y[fold], y_pred_prob, ids[fold])
                df["FOLD"] = fold
                combined_dfs.append(df)
                export_prob_hist(df.copy(), out_dir.joinpath(f"probabilities_fold{fold}"))
            _save_and_preview_predictions(
                pd.concat(combined_dfs, ignore_index=True),
                out_dir.joinpath("predictions_cv.tsv"),
            )

    else:
        _export_prediction_data(out_dir, algorithm, y, x, ids, y_train=y_train, x_train=x_train, ids_train=ids_train)


def _export_prediction_data(out_dir, algorithm, y, x, ids, y_train=None, x_train=None, ids_train=None, fold=None):
    """
    Save probability histograms and tables with accuracy metrics for a single fold or non-CV run.

    Args:
        out_dir (pathlib.Path): Directory where results are saved.
        algorithm: Discrete prediction algorithm.
        y (pandas.DataFrame): Ground truth phenotypes.
        x (pandas.DataFrame): Input data.
        ids (pandas.Series): Participant IDs.
        y_train (optional, pandas.DataFrame): Training ground truth (Default: None).
        x_train (optional, pandas.DataFrame): Training input data (Default: None).
        ids_train (optional, pandas.Series): Training participant IDs (Default: None).
        fold (int): CV fold number; None or 0 omits the fold suffix (Default: None).
    """

    suffix = f"_fold{fold}" if fold not in (None, 0) else ""

    y_pred_prob = algorithm.predict_proba(x)
    if x_train is not None:
        y_train_pred = algorithm.predict_proba(x_train)

    if y_train is not None and y_train_pred is not None and ids_train is not None:
        export_prediction_tables(
            y_train,
            y_train_pred,
            ids_train,
            out_dir.joinpath(f"train_predictions{suffix}.tsv"),
            dataset="training",
        )

    df_prediction = export_prediction_tables(
        y,
        y_pred_prob,
        ids,
        out_dir.joinpath(f"predictions{suffix}.tsv"),
    )

    export_prob_hist(
        df_prediction,
        out_dir.joinpath(f"probabilities{suffix}"),
    )


def additional_sumstats(algorithm_name, y_test, x_test, algorithm, prefix, is_using_outer_cv, is_testing=False):
    if is_using_outer_cv:
        if is_testing:
            rows = []
            for fold, y_test_fold in enumerate(y_test):
                y_pred_fold = algorithm[fold].predict_proba(x_test[fold])
                row = _build_sumstats_df(algorithm_name, y_test_fold, y_pred_fold)
                row["FOLD"] = fold + 1
                rows.append(row)
            combined = pd.concat(rows, ignore_index=True)
            combined.to_csv(prefix.joinpath("performance_metrics.tsv"), index=False, sep="\t")
        else:
            # Fold 0 is the full-data fit — write it on its own.
            y_pred_full = algorithm[0].predict_proba(x_test[0])
            _build_sumstats_df(algorithm_name, y_test[0], y_pred_full).to_csv(
                prefix.joinpath("performance_metrics.tsv"), index=False, sep="\t"
            )
            # Folds 1-N combined into the CV metrics file.
            rows = []
            for fold in range(1, len(algorithm)):
                y_pred_fold = algorithm[fold].predict_proba(x_test[fold])
                row = _build_sumstats_df(algorithm_name, y_test[fold], y_pred_fold)
                row["FOLD"] = fold
                rows.append(row)
            combined = pd.concat(rows, ignore_index=True)
            combined.to_csv(prefix.joinpath("performance_metrics_cv.tsv"), index=False, sep="\t")
    else:
        y_pred = algorithm.predict_proba(x_test)
        _build_sumstats_df(algorithm_name, y_test, y_pred).to_csv(
            prefix.joinpath("performance_metrics.tsv"), index=False, sep="\t"
        )


def _build_sumstats_df(algorithm_name, y_test, y_pred):
    """
    Build a single-row performance metrics dataframe without saving it.

    Args:
        algorithm_name (str): Classifier model name.
        y_test (numpy.ndarray): Ground truth phenotypes.
        y_pred (numpy.ndarray): Raw output of predict_proba (both columns).

    :return: *(pandas.DataFrame)*: \n
        One-row dataframe with columns Algorithm, AUC, Accuracy, Balanced_Accuracy,
        Log_Loss, Sensitivity, Specificity, PPV, NPV.
    """
    return pd.DataFrame(
        data=[[algorithm_name] + list(_calculate_accuracy_scores(y_test, y_pred))],
        columns=["Algorithm", "AUC", "Accuracy", "Balanced_Accuracy", "Log_Loss", "Sensitivity", "Specificity", "PPV", "NPV"],
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

    y_pred_prob = algorithm.predict_proba(x)
    return _calculate_accuracy_scores(y, y_pred_prob)


def _calculate_accuracy_scores(y, y_pred_prob):
    """
    Calculate accuracy metrics for the chosen discrete prediction model.

    Args:
        y (pandas.DataFrame): Reported output features.
        y_pred_prob (pandas.DataFrame): Predicted case probabilities.
    
    :return: rocauc *(float)*: \n
        ROC AUC value.
    :return: acc *(float)*: \n
        Accuracy value.
    :return: balacc *(float)*: \n
        Balanced accuracy value.
    :return: ll *(float)*: \n
        Log loss value.
    :return: sens *(float)*: \n
        Sensitivity value.
    :return: spec *(float)*: \n
        Specificity value.
    :return: ppv *(float)*: \n
        Positive predictive value.
    :return: npv *(float)*: \n
        Negative predictive value.
    """

    y_pred = np.argmax(y_pred_prob, axis=1)
    y_pred_prob = y_pred_prob[:,1]

    rocauc = metrics.roc_auc_score(y, y_pred_prob)
    acc = metrics.accuracy_score(y, y_pred) * 100
    balacc = metrics.balanced_accuracy_score(y, y_pred) * 100
    ll = metrics.log_loss(y, y_pred_prob)
    
    cm = metrics.confusion_matrix(y, y_pred)
    tn = cm[0][0]
    fn = cm[1][0]
    tp = cm[1][1]
    fp = cm[0][1]
    sens = (tp / (tp + fn) if (tp + fn) > 0 else 0)
    spec = (tn / (tn + fp) if (tn + fp) > 0 else 0)
    ppv  = (tp / (tp + fp) if (tp + fp) > 0 else 0)
    npv  = (tn / (tn + fn) if (tn + fn) > 0 else 0)

    return rocauc, acc, balacc, ll, sens, spec, ppv, npv


def _build_prediction_df(y, y_pred_prob, ids):
    """
    Construct a prediction dataframe without saving or printing anything.

    Args:
        y (pandas.DataFrame): Ground truth phenotypes.
        y_pred_prob (numpy.ndarray): Raw output of predict_proba (both columns).
        ids (pandas.Series): Participant IDs.

    :return: df_prediction *(pandas.DataFrame)*: \n
        Table with columns ID, CASE_REPORTED, CASE_PROBABILITY, CASE_PREDICTED.
    """
    y_pred_df = pd.DataFrame(y_pred_prob, dtype=float)
    df_predicted_cases = y_pred_df.idxmax(axis=1)
    case_probs = pd.DataFrame(y_pred_df.iloc[:, 1])
    ids_df = pd.DataFrame(ids)
    df_prediction = pd.concat(
        [
            ids_df.reset_index(drop=True),
            y.reset_index(drop=True),
            case_probs.reset_index(drop=True),
            df_predicted_cases.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )
    df_prediction.columns = ["ID", "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
    return df_prediction


def _save_and_preview_predictions(df_prediction, output_path, dataset="withheld test"):
    """
    Save a prediction dataframe to disk and print a preview.

    Args:
        df_prediction (pandas.DataFrame): Prediction table to save.
        output_path (pathlib.Path): Destination file path.
        dataset (str): Label used in the preview message.
    """
    df_prediction.to_csv(output_path, index=False, sep="\t")
    print("")
    print(f"Preview of the exported predictions for the {dataset} data that has been exported as {output_path}.")
    print("")
    print("#" * 70)
    print(df_prediction.head())
    print("#" * 70)


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

    df_prediction = _build_prediction_df(y, y_pred, ids)
    _save_and_preview_predictions(df_prediction, output_path, dataset=dataset)
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
