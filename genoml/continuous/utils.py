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
import statsmodels.formula.api as sm
from itertools import repeat
from genoml import utils
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.metrics import mean_squared_error


def export_prediction_data(out_dir, ids, step, algorithm, y, x, is_using_outer_cv, y_withheld=None, x_withheld=None, ids_withheld=None):
    if is_using_outer_cv:
        # Store prediction results for each fold
        results_cv = []
        for fold, algo in enumerate(algorithm):
            y_predicted = algo.predict(x[fold])
            if fold == 0 and step != "testing":
                results = pd.DataFrame(
                    zip(ids[fold], y[fold], y_predicted), 
                    columns=["ID", "REPORTED", "PREDICTED"],
                )
            else:
                results_cv.append(pd.DataFrame(
                    zip(ids[fold], y[fold], y_predicted, repeat(fold)), 
                    columns=["ID", "REPORTED", "PREDICTED", "FOLD"],
                ))

        # Combine all folds
        results_cv = pd.concat(results_cv, ignore_index=True)
            
    else:
        y_predicted = algorithm.predict(x)
        if x_withheld is not None:
            y_withheld_predicted = algorithm.predict(x_withheld)

        # Training results.
        results = pd.DataFrame(
            zip(ids, y, y_predicted), 
            columns=["ID", "REPORTED", "PREDICTED"],
        )
        if step == "training":
            withheld_results = pd.DataFrame(
                zip(ids_withheld, y_withheld, y_withheld_predicted), 
                columns=["ID", "REPORTED", "PREDICTED"],
            )
    
    if step != "testing" or (step == "testing" and not is_using_outer_cv):
        with utils.DescriptionLoader.context(
            "utils/export_predictions",
            output_path=out_dir.joinpath(f"{step}_predictions.tsv"), 
            data=results.head(),
        ):
            results.to_csv(
                out_dir.joinpath(f"{step}_predictions.tsv"), 
                index=False, 
                sep="\t",
            )

        _plot_results(
            out_dir.joinpath(f"regression_summary.txt"),
            out_dir.joinpath(f"regression.png"), 
            results,
            False,
        )

    if is_using_outer_cv:
        with utils.DescriptionLoader.context(
            "utils/export_predictions",
            output_path=out_dir.joinpath(f"{step}_predictions_cv.tsv"), 
            data=results_cv.head(),
        ):
            results_cv.to_csv(
                out_dir.joinpath(f"{step}_predictions_cv.tsv"), 
                index=False, 
                sep="\t",
            )

        _plot_results(
            out_dir.joinpath(f"regression_summary_cv.txt"),
            out_dir.joinpath(f"regression_cv.png"), 
            results_cv,
            True,
        )

    if step == "training":
        with utils.DescriptionLoader.context(
            "utils/export_predictions/withheld_data",
            output_path=out_dir.joinpath(f"{step}_predictions_withheld.tsv"), 
            data=withheld_results.head(),
        ):
            withheld_results.to_csv(
                out_dir.joinpath(f"{step}_predictions_withheld.tsv"), 
                index=False, 
                sep="\t",
            )

        _plot_results(
            out_dir.joinpath(f"regression_summary_withheld.txt"),
            out_dir.joinpath("regression_withheld.png"), 
            withheld_results,
            False,
        )


def _plot_results(regression_summary_path, regression_plot_path, df, is_using_outer_cv):
    plt.figure()

    # Fit overall regression line and add it to the plot
    reg_model = sm.ols("REPORTED ~ PREDICTED", data=df).fit()

    with utils.DescriptionLoader.context(
        "utils/export_predictions/plot",
        output_path=regression_plot_path, 
        data=reg_model.summary(),
    ):
        # Plot reported vs predicted value for each sample
        if is_using_outer_cv:
            sns.scatterplot(
                data=df,
                x="REPORTED",
                y="PREDICTED",
                style="FOLD",
                hue="FOLD",
                palette="tab10",
            )
        else:
            sns.scatterplot(
                data=df,
                x="REPORTED",
                y="PREDICTED",
                palette="tab10",
            )
        
        with open(regression_summary_path, "w") as f:
            f.write(reg_model.summary().as_csv().replace(",", "\t"))
        sns.regplot(
            data=df,
            x="REPORTED",
            y="PREDICTED",
            scatter=False,
            color="black",
            label="Best-fit trendline",
        )

        # Add dashed line at y=x
        ax = plt.gca()
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(
            lims, 
            lims, 
            'k--', 
            alpha=0.5, 
            zorder=0, 
            label="Predicted = Reported",
        )

        pearson_r, _ = pearsonr(df["REPORTED"].values, df["PREDICTED"].values)
        rmse = np.sqrt(mean_squared_error(df["REPORTED"], df["PREDICTED"]))
        plt.text(
            0.95, 0.05,
            f"$r = {pearson_r:.3f}$\n$RMSE = {rmse:.2f}$",
            ha="right",
            va="bottom",
            transform=plt.gca().transAxes,
            fontsize=14,
        )

        plt.legend(fontsize=8)

        plt.savefig(regression_plot_path, dpi=600)
        plt.clf()


def additional_sumstats(algorithm_name, y_test, x_test, algorithm, prefix, is_using_outer_cv):
    if is_using_outer_cv:
        y_true = []
        y_pred = []
        for fold, y_test_fold in enumerate(y_test):
            y_pred_fold = algorithm[fold].predict(x_test[fold])
            y_true.append(y_test_fold)
            y_pred.append(y_pred_fold)
        y_true = np.concatenate([np.array(y) for y in y_true])
        y_pred = np.concatenate([np.array(y) for y in y_pred])
    else:
        y_true = y_test.copy()
        y_pred = algorithm.predict(x_test)
    _additional_sumstats(algorithm_name, y_true, y_pred, prefix)


def _additional_sumstats(algorithm_name, y_test, y_pred, prefix):
    log_table = pd.DataFrame(
        data=[[algorithm_name] + list(_calculate_accuracy_scores(y_test, y_pred))], 
        columns=["Algorithm", "Explained Variance", "Root Mean Squared Error", "Median Absolute Error", "Pearson's R"],
    )
    log_outfile = prefix.joinpath(f"performance_metrics.tsv")
    log_table.to_csv(log_outfile, index=False, sep="\t")


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

    :return: evs *(float)*: \n
        Explained Variance Score.
    :return: rmse *(float)*: \n
        Root Mean Squared Error.
    :return: mae *(float)*: \n
        Median Absolute Error.
    :return: pearson_r *(float)*: \n
        Pearson's R.
    """

    evs = metrics.explained_variance_score(y, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y, y_pred))
    mae = metrics.median_absolute_error(y, y_pred)
    pearson_r, _ = pearsonr(y, y_pred)

    return evs, rmse, mae, pearson_r
