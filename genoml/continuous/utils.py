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
import seaborn as sns
import statsmodels.formula.api as sm
from sklearn import metrics
from genoml import utils


def export_prediction_data(out_dir, ids, step, y, y_predicted, y_withheld=None, y_withheld_predicted=None):
    """
    Save table with predicted vs. reported phenotypes for each sample and generate regression model.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        ids (numpy.ndarray): Array of sample IDs.
        step (str): Step of the GenoML workflow. 
        y (numpy.ndarray): Array of reported phenotypes for each training sample.
        y_predicted (numpy.ndarray): Array of predicted phenotypes for each training sample.
        y_withheld (numpy.ndarray, optional): Array of reported phenotypes for each validation sample (Default: None).
        y_withheld_predicted (numpy.ndarray, optional): Array of predicted phenotypes for each validation sample (Default: None).

    :return: results *(pandas.DataFrame)*: \n
        Table with predicted and reported phenotypes.
    """

    output_columns = ["ID", f"REPORTED", f"PREDICTED"]

    # Training results.
    results = pd.DataFrame(
        zip(ids, y, y_predicted), 
        columns=output_columns,
    )
    with utils.DescriptionLoader.context(
        "utils/export_predictions",
        output_path=out_dir.joinpath(f'{step}_predictions.txt'), 
        data=results.head(),
    ):
        results.to_csv(out_dir.joinpath(f'{step}_predictions.txt'), index=False, sep="\t")

    # Withheld results, if applicable.
    if step == "training":
        results = pd.DataFrame(
            zip(ids, y_withheld, y_withheld_predicted), 
            columns=output_columns,
        )
        with utils.DescriptionLoader.context(
            "utils/export_predictions/withheld_data",
            output_path=out_dir.joinpath('withheld_predictions.txt'), 
            data=results.head(),
        ):
            results.to_csv(out_dir.joinpath('withheld_predictions.txt'), index=False, sep="\t")

    # Regression model on withheld results.
    reg_model = sm.ols(formula=f'REPORTED ~ PREDICTED', data=results)
    fitted = reg_model.fit()
    with utils.DescriptionLoader.context(
        "utils/export_predictions/plot",
        output_path=out_dir.joinpath('regression.png'), 
        data=fitted.summary(),
    ):
        sns_plot = sns.regplot(
            data=results, 
            x=f"REPORTED", 
            y=f"PREDICTED", 
            scatter_kws={"color": "blue"},
            line_kws={"color": "red"},
        )
        plt.text(
            0.95, 0.05,
            f"$R^2 = {fitted.rsquared:.3f}$",
            ha="right", 
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=18,
        )
        sns_plot.set_xlabel(f"Reported", fontsize=16)
        sns_plot.set_ylabel(f"Predicted", fontsize=16)

        sns_plot.figure.savefig(out_dir.joinpath('regression.png'), dpi=600)
        with open(out_dir.joinpath('regression_summary.txt'), "w") as f:
            f.write(fitted.summary().as_csv().replace(",", "\t"))


def performance_metrics(out_dir, y_reported, y_predicted):
    """
    Evaluate and save performance metrics.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        y_reported (numpy.ndarray): Array of reported phenotypes for each sample.
        y_predicted (numpy.ndarray): Array of predicted phenotypes for each sample.
    """

    evs = metrics.explained_variance_score(y_reported, y_predicted)
    mse = metrics.mean_squared_error(y_reported, y_predicted)
    mae = metrics.median_absolute_error(y_reported, y_predicted)
    r2s = metrics.r2_score(y_reported, y_predicted)

    print("")
    print("#"*70)
    print(f"Explained variance score: {evs:.4}")
    print(f"Mean squared error: {mse:.4}")
    print(f"Median absolute error: {mae:.4}")
    print(f"R^2 score: {r2s:.4}")
        
    log_entry = pd.DataFrame({
        "Explained_variance_score": [evs],
        "Mean_squared_error": [mse], 
        "Median_absolute_error": [mae], 
        "R2_score": [r2s]
    })

    log_outfile = out_dir.joinpath('performance_metrics.txt')
    log_entry.to_csv(log_outfile, index=False, sep="\t")

    print("#"*70)
    print("")
    print(f"This table below is also logged as {log_outfile} and is in your current working directory...")
    print("#"*70)
    print(log_entry)
    print("#"*70)
    print("")


def calculate_accuracy_scores(x, y, algorithm):
    """
    Calculate accuracy metrics for the chosen continuous prediction model.

    Args:
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        algorithm: Contonuous prediction algorithm.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the continuous prediction module.
    """

    y_pred = algorithm.predict(x)
    return _calculate_accuracy_scores(y, y_pred)


def _calculate_accuracy_scores(y, y_pred):
    """
    Calculate accuracy metrics for the chosen continuous prediction model.

    Args:
        y (pandas.DataFrame): Reported output features.
        y_pred (pandas.DataFrame): Predicted output features.

    :return: accuracy_metrics *(list)*: \n
        Accuracy metrics used for the continuous prediction module.
    """

    evs = metrics.explained_variance_score(y, y_pred)
    mse = metrics.mean_squared_error(y, y_pred)
    mae = metrics.median_absolute_error(y, y_pred)
    r2s = metrics.r2_score(y, y_pred)

    accuracy_metrics = [evs, mse, mae, r2s]
    return accuracy_metrics
