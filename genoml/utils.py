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

import copy
import joblib
import json
import numpy as np
import os
import pandas as pd
import sys
import textwrap
import time
import traceback
from pathlib import Path
from sklearn import model_selection
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

# os.environ["PYTHONWARNINGS"] = "ignore"
# sys.stderr = open(os.devnull, 'w')

__author__ = 'Sayed Hadi Hashemi'


class ColoredBox:
    BLACK = 30
    RED = 31
    GREEN = 32
    YELLOW = 33
    BLUE = 34
    MAGENTA = 35
    CYAN = 36
    WHITE = 37
    RESET = 39

    def __init__(self, color=None):
        if color is None:
            color = self.GREEN
        self.__color = color

    def __enter__(self):
        print('\033[{}m'.format(self.__color), end="")

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("\x1b[0m", end="")

    @classmethod
    def wrap(cls, text, color):
        return '\033[{}m'.format(color) + text + "\x1b[0m"


class ContextScope:
    indent = 0
    _verbose = False

    def __init__(self, title, description, error, start=True, end=False,
                 **kwargs):
        self._title = title.format(**kwargs)
        self._description = description.format(**kwargs)
        self._error = error.format(**kwargs)
        self._start = start
        self._end = end

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and exc_val is None and exc_tb is None:
            if self._end:
                print(
                    "{}{}: {}".format(
                        self.get_prefix(ColoredBox.GREEN),
                        ColoredBox.wrap(self._title, ColoredBox.GREEN),
                        ColoredBox.wrap('[Done]', ColoredBox.GREEN)))
            self.remove_indent()
        else:
            print("{}{}: {}".format(
                self.get_prefix(ColoredBox.RED), self._title,
                ColoredBox.wrap('[Failed]', ColoredBox.RED)))
            print("{}".format(self.indent_text(self._error)))
            self.remove_indent()
            traceback.print_exception(exc_type, exc_val, exc_tb)
            exit(1)

    def __enter__(self):
        self.add_indent()
        if self._start:
            print()
            print("{}{}".format(self.get_prefix(ColoredBox.BLUE),
                                ColoredBox.wrap(self._title, ColoredBox.BLUE)))
        if self._verbose and self._description:
            print("{}".format(self._description))

    @classmethod
    def add_indent(cls):
        cls.indent += 1

    @classmethod
    def remove_indent(cls):
        cls.indent -= 1

    @classmethod
    def get_prefix(cls, color=None):
        indent_size = 4
        text = "---> " * cls.indent
        if color:
            text = ColoredBox.wrap(text, color)
        return text

    @classmethod
    def indent_text(cls, text):
        WIDTH = 70
        indent = max(0, len(cls.get_prefix()) - 2)
        width = WIDTH - indent
        ret = textwrap.fill(text, width)
        ret = textwrap.indent(ret, " " * indent)
        return ret

    @classmethod
    def set_verbose(cls, verbose):
        cls._verbose = verbose


class DescriptionLoader:
    _descriptions = None

    @classmethod
    def _load(cls):
        description_file = os.path.join(os.path.dirname(__file__),
                                        "misc", "descriptions.json")
        with open(description_file) as fp:
            cls._descriptions = json.load(fp)

    @classmethod
    def function_description(cls, key, **kwargs):
        dkwargs = cls.get(key)
        return function_description(**dkwargs, **kwargs)

    @classmethod
    def get(cls, key):
        if cls._descriptions is None:
            cls._load()
        return cls._descriptions[key]

    @classmethod
    def context(cls, key, **kwargs):
        dkwargs = cls.get(key)
        return ContextScope(**dkwargs, **kwargs)

    @classmethod
    def print(cls, key, **kwargs):
        dkwargs = cls.get(key)
        with ContextScope(**dkwargs, **kwargs):
            pass


class Timer:
    def __init__(self):
        self.start = None
        self.end = None

    def start_timer(self):
        self.start = time.time()

    def __enter__(self):
        self.start_timer()
        return self

    def __exit__(self, *args):
        self.stop_timer()

    def stop_timer(self):
        self.end = time.time()

    def elapsed(self):
        return self.end - self.start


def function_description(**dkwargs):
    def wrap(func):
        def func_wrapper(*args, **kwargs):
            with ContextScope(**dkwargs):
                return func(*args, **kwargs)
        return func_wrapper
    return wrap


def metrics_to_str(metrics_dict):
    """
    Convert training metrics to string.

    Args:
        metrics_dict (dict): Metric names and corresponding values.

    :return: metrics_str *(str)*: \n
        Description of accuracy metrics.
    """

    rows = []
    for key, value in metrics_dict.items():
        if key == "Algorithm":
            rows.append("{}: {}".format(key, value))
        elif key == "Runtime_Seconds":
            rows.append("{}: {:0.3f} seconds\n".format(key, value))
        else:
            rows.append("{}: {:0.4f}".format(key, value))
    
    metrics_str = str.join("\n", rows)
    return metrics_str


def create_results_dir(prefix, module):
    """
    Create output directory for the given GenoML module.

    Args:
        prefix (pathlib.Path): Path to output directory.
        module (str): GenoML module being used.

    :return: results_path *(pathlib.Path)*: \n
        Path to results directory.
    """

    prefix = Path(prefix)
    if not prefix.is_dir():
        prefix.mkdir()
        
    results_path = prefix.joinpath(module)
    if not results_path.is_dir():
        results_path.mkdir()
    return results_path


def select_best_algorithm(log_table, metric_max, algorithms):
    """
    Choose the best-performing algorithm based on the provided criteria.

    Args:
        log_table (pandas.DataFrame): Results for each trained model.
        metric_max (str): Indicator for the metric used to compare algorithm performance.
        algorithms (dict): Names and corresponding functions for each algorithm being used for training.
    
    :return: best_algorithm: \n
        Best-perorming algorithm based on the indicated criteria.
    """

    best_id = log_table[metric_max].idxmax()
    best_algorithm_name = log_table.loc[best_id, "Algorithm"]
    best_algorithm = algorithms[best_algorithm_name]
    best_algorithm_metrics = log_table.loc[best_id].to_dict()

    DescriptionLoader.print(
        "utils/training/compete/algorithm/best",
        algorithm=best_algorithm_name,
        metrics=metrics_to_str(best_algorithm_metrics),
    )

    return best_algorithm, best_algorithm_name


def tune_model(estimator, x, y, param_distributions, scoring, n_iter, cv, random_state, is_using_outer_cv):
    """
    Apply randomized search to fine-tune the selected model.

    Args:
        estimator: Trained baseline model.
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        param_distributions (dict): Hyperparameters and corresponsing values to be tested.
        scoring (sklearn.metrics._scorer._Scorer): Scoring metric to evaluate accuracy.
        n_iter (int): Maximum number of iterations.
        cv (sklearn.model_selection (Stratified)KFold): Cross-validation generator.
        random_state (int): Random seed for reproducibility.
    
    :return: cv_results *(dict)*: \n
        Results from hyperparameter tuning.
    :return: algo_tuned: \n
        Tuned model.
    """

    if is_using_outer_cv:
        cv_results = []
        algo_tuned = []
        for fold in range(len(estimator)):
            cv_results_fold, algo_tuned_fold = _tune_model(estimator[fold], x[fold], y[fold], param_distributions, scoring, n_iter, cv, random_state)
            cv_results.append(cv_results_fold)
            algo_tuned.append(algo_tuned_fold)
    else:
        cv_results, algo_tuned = _tune_model(estimator, x, y, param_distributions, scoring, n_iter, cv, random_state)

    return cv_results, algo_tuned


def _tune_model(estimator, x, y, param_distributions, scoring, n_iter, cv, random_state):
    """
    Apply randomized search to fine-tune the selected model.

    Args:
        estimator: Trained baseline model.
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        param_distributions (dict): Hyperparameters and corresponsing values to be tested.
        scoring (sklearn.metrics._scorer._Scorer): Scoring metric to evaluate accuracy.
        n_iter (int): Maximum number of iterations.
        cv (sklearn.model_selection (Stratified)KFold): Cross-validation generator.
        random_state (int): Random seed for reproducibility.
    
    :return: cv_results *(dict)*: \n
        Results from hyperparameter tuning.
    :return: algo_tuned: \n
        Tuned model.
    """
    rand_search = BayesSearchCV(
        estimator = estimator,
        search_spaces = param_distributions,
        scoring = scoring,
        n_iter = n_iter,
        cv = cv,
        n_jobs = -1,
        random_state = random_state,
        verbose = 0,
        error_score = np.nan
    )

    with Timer() as timer:
        try:
            rand_search.fit(x, y)
        except Exception as e:
            print(f"ERROR during BayesSearchCV fit: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            traceback.print_exc()
            raise
    print(f"BayesSearchCV took {timer.elapsed():.2f} seconds for {n_iter:d} "
          "candidate parameter iterations.")

    cv_results = rand_search.cv_results_
    algo_tuned = rand_search.best_estimator_
    return cv_results, algo_tuned


def summarize_tune(out_dir, estimator_baseline, estimator_tune, x, y, scoring, cv, is_using_outer_cv):
    """
    Use cross-validation to compare the tuned model to the trined 
    baseline model. 

    Args:
        out_dir (pathlib.Path): Path to output directory.
        estimator_baseline: Trained baseline model.
        estimator_tune: Tuned model.
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        scoring (sklearn.metrics._scorer._Scorer): Scoring metric to evaluate accuracy.
        cv (sklearn.model_selection (Stratified)KFold): Cross-validation generator.

    :return: cv_baseline *(pandas.DataFrame)*: \n
        Cross-validation results for the trained baseline model.
    :return: cv_tuned *(pandas.DataFrame)*: \n
        Cross-validation results for the tuned model.
    """

    if is_using_outer_cv:
        cv_baseline = [] 
        cv_tuned = []
        for fold in range(len(estimator_baseline)):
            cv_baseline_fold, cv_tuned_fold = _summarize_tune(out_dir, estimator_baseline[fold], estimator_tune[fold], x[fold], y[fold], scoring, cv, fold=fold)
            cv_baseline.append(cv_baseline_fold)
            cv_tuned.append(cv_tuned_fold)

    else:
        cv_baseline, cv_tuned = _summarize_tune(out_dir, estimator_baseline, estimator_tune, x, y, scoring, cv)

    return cv_baseline, cv_tuned


def _summarize_tune(out_dir, estimator_baseline, estimator_tune, x, y, scoring, cv, fold=None):
    """
    Use cross-validation to compare the tuned model to the trined 
    baseline model. 

    Args:
        out_dir (pathlib.Path): Path to output directory.
        estimator_baseline: Trained baseline model.
        estimator_tune: Tuned model.
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        scoring (sklearn.metrics._scorer._Scorer): Scoring metric to evaluate accuracy.
        cv (sklearn.model_selection (Stratified)KFold): Cross-validation generator.
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).

    :return: cv_baseline *(pandas.DataFrame)*: \n
        Cross-validation results for the trained baseline model.
    :return: cv_tuned *(pandas.DataFrame)*: \n
        Cross-validation results for the tuned model.
    """

    suffix = f"_fold{fold}" if fold not in (None, 0) else ""

    cv_baseline = model_selection.cross_val_score(
        estimator = estimator_baseline, 
        X = x, 
        y = y, 
        scoring = scoring, 
        cv = cv, 
        n_jobs = -1, 
        verbose = 0,
    )

    cv_tuned = model_selection.cross_val_score(
        estimator = estimator_tune, 
        X = x, 
        y = y, 
        scoring = scoring, 
        cv = cv, 
        n_jobs = -1, 
        verbose = 0,
    )

    # Output a log table summarizing CV mean scores and standard deviations
    df_cv_summary = pd.DataFrame({
        "Mean_CV_Score" : [cv_baseline.mean(), cv_tuned.mean()],
        "Standard_Dev_CV_Score" : [cv_baseline.std(), cv_tuned.std()],
        "Min_CV_Score" : [cv_baseline.min(), cv_tuned.min()],
        "Max_CV_Score" : [cv_baseline.max(), cv_tuned.max()],
    })
    df_cv_summary.rename(index={0: "Baseline", 1: "BestTuned"}, inplace=True)
    log_outfile = out_dir.joinpath(f"cv_summary{suffix}.tsv")
    df_cv_summary.to_csv(log_outfile, sep="\t")

    print(f"Here is the cross-validation summary of your best tuned model hyperparameters{'' if fold is None else (' for the full dataset' if fold==0 else f' for fold {fold+1}')}...")
    print(f"{scoring} scores per cross-validation")
    print(cv_tuned)
    print(f"Mean cross-validation score:                        {cv_tuned.mean()}")
    print(f"Standard deviation of the cross-validation score:   {cv_tuned.std()}")
    print("")
    print("Here is the cross-validation summary of your baseline/default hyperparamters for "
          "the same algorithm on the same data...")
    print(f"{scoring} scores per cross-validation")
    print(cv_baseline)
    print(f"Mean cross-validation score:                        {cv_baseline.mean()}")
    print(f"Standard deviation of the cross-validation score:   {cv_baseline.std()}")
    print("")
    print("Just a note, if you have a relatively small variance among the cross-validation "
          "iterations, there is a higher chance of your model being more generalizable to "
          "similar datasets.")
    print(f"We are exporting a summary table of the cross-validation mean score and standard "
          f"deviation of the baseline vs. best tuned model here {log_outfile}.")

    return cv_baseline, cv_tuned


def report_best_tuning(out_dir, cv_results, n_top, is_using_outer_cv):
    """
    Find the top-performing tuning iterations and save those to a table.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_results (dict): Results from hyperparameter tuning.
        n_top (int): Number of iterations to report.
    """

    if is_using_outer_cv:
        for fold in range(len(cv_results)):
            _report_best_tuning(out_dir, cv_results[fold], n_top, fold=fold)
    else:
        _report_best_tuning(out_dir, cv_results, n_top)


def _report_best_tuning(out_dir, cv_results, n_top, fold=None):
    """
    Find the top-performing tuning iterations and save those to a table.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_results (dict): Results from hyperparameter tuning.
        n_top (int): Number of iterations to report.
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    """

    suffix = f"_fold{fold}" if fold not in (None, 0) else ""

    print(f"Here is a summary of the top 10 iterations of the hyperparameter tuning{'' if fold is None else (' for the full dataset' if fold==0 else f' for fold {fold+1}')}...")
    cv_results = pd.DataFrame(cv_results)
    cv_results.sort_values(by='rank_test_score', ascending=True, inplace=True)
    cv_results = cv_results.iloc[:n_top,:]
    for i in range(len(cv_results)):
        current_iteration = cv_results.iloc[i,:]
        print(f"Model with Rank {i + 1}:")
        print(f"Mean Validation Score: {current_iteration['mean_test_score']:.3f} (std: {current_iteration['std_test_score']:.3f})")
        print(f"Parameters: {current_iteration['params']}")
        print("")
    log_outfile = out_dir.joinpath(f"tuning_summary{suffix}.tsv")
    cv_results.to_csv(log_outfile, index=False, sep="\t")
    print(f"We are exporting a summary table of the top {n_top} iterations of the hyperparameter tuning step and its parameters here {log_outfile}.")


def compare_tuning_performance(out_dir, cv_tuned, cv_baseline, algo_tuned, algo_baseline, is_using_outer_cv, x=None):
    """
    Determine whether the tuned model outperformed the baseline model.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_tuned (pandas.DataFrame): Cross-validation results for the tuned model.
        cv_baseline (pandas.DataFrame): Cross-validation results for the trained baseline model.
        algo_tuned: Tuned model.
        algo_baseline: Trained baseline model.
        x (pandas.DataFrame, optional): Model input features (Default: None).
    
    :return: algorithm: \n
        Better-perorming of the tuned and trained baseline models.
    :return: y_predicted *(numpy.ndarray)*: \n
        Predicted outputs from the chosen model.
    """

    if is_using_outer_cv:
        algorithm = []
        y_predicted = []
        for fold in range(len(cv_tuned)):
            algorithm_fold, y_predicted_fold = _compare_tuning_performance(out_dir, cv_tuned[fold], cv_baseline[fold], algo_tuned[fold], algo_baseline[fold], x=x[fold], fold=fold)
            algorithm.append(algorithm_fold)
            y_predicted.append(y_predicted_fold)
    else:
        algorithm, y_predicted = _compare_tuning_performance(out_dir, cv_tuned, cv_baseline, algo_tuned, algo_baseline, x=x)

    return algorithm, y_predicted


def _compare_tuning_performance(out_dir, cv_tuned, cv_baseline, algo_tuned, algo_baseline, x=None, fold=None):
    """
    Determine whether the tuned model outperformed the baseline model.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_tuned (pandas.DataFrame): Cross-validation results for the tuned model.
        cv_baseline (pandas.DataFrame): Cross-validation results for the trained baseline model.
        algo_tuned: Tuned model.
        algo_baseline: Trained baseline model.
        x (pandas.DataFrame, optional): Model input features (Default: None).
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    
    :return: algorithm: \n
        Better-perorming of the tuned and trained baseline models.
    :return: y_predicted *(numpy.ndarray)*: \n
        Predicted outputs from the chosen model.
    """

    if fold == 0:
        if cv_baseline.mean() >= cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
            print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
            algorithm = algo_baseline
            yield algorithm

        if cv_baseline.mean() < cv_tuned.mean():
            print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
            print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
            algorithm = algo_tuned
            yield algorithm
    
    else:
        algorithm = algo_tuned
        yield algorithm

    export_model(out_dir.parent, algorithm, fold=fold)

    y_predicted = None
    if x is not None:
        y_predicted = algorithm.predict(x)
    yield y_predicted


def read_munged_data(file_path):
    """
    Read munged hdf5 file to pandas

    Args:
        out_dir (pathlib.Path): Path to output directory.
    
    :return: df: \n
        Munged dataset.
    """

    with DescriptionLoader.context(
        "read_munge", 
        path=file_path,
    ):
        df = pd.read_hdf(file_path, key="dataForML")

    DescriptionLoader.print(
        "data_summary", 
        data=df.describe(),
    )

    return df


def export_model(out_dir, algorithm, fold=None):
    """
    Export a fitted algorithm to a readable file.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        algorithm: Fitted algorithm being exported.
        fold (int): If using outer cross-validation, fold number corresponding to current data/algorithm (Default: None).
    """

    if fold not in (0, None):
        output_path = out_dir.joinpath(f"model_fold{fold}.joblib")
    else:
        output_path = out_dir.joinpath("model.joblib")
    print(output_path)
    with DescriptionLoader.context(
        "export_model",
        output_path=output_path,
    ):
        joblib.dump(algorithm, output_path)


@DescriptionLoader.function_description("utils/training/compete")
def fit_algorithms(out_dir, algorithms, x_train, y_train, x_valid, y_valid, column_names, calculate_accuracy_scores):
    """
    Compete algorithms against each other during the training stage and record results.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        algorithms (dict): Names and corresponding functions for each algorithm being used for training.
        x_train (pandas.DataFrame): Model input features for training.
        y_train (pandas.DataFrame): Reported output features for training.
        x_valid (pandas.DataFrame): Model input features for validation.
        y_valid (pandas.DataFrame): Reported output features for validation.
        column_names (list): Names for each feature, to serve as column headers in resulting data table.
        calculate_accuracy_scores (func): Function for accuracy score calculation for the given module.
    
    :return: log_table: \n
        Results for each trained model.
    """

    log_table = []
    trained_algorithms = dict()

    for algorithm_name, algorithm in algorithms.items():
        with DescriptionLoader.context(
            "utils/training/fit_algorithms/compete/algorithm",
            name=algorithm_name,
        ):
            trained_algorithms[algorithm_name] = []
            algorithm_log, trained_algorithm = _fit_algorithm(algorithm, algorithm_name, x_train, y_train, x_valid, y_valid, column_names, calculate_accuracy_scores)
            log_table.append(algorithm_log)
            trained_algorithms[algorithm_name] = trained_algorithm

    log_table = pd.DataFrame(data=log_table, columns=column_names)
    output_path = out_dir.joinpath('withheld_performance_metrics.tsv')
    with DescriptionLoader.context(
        "utils/training/fit_algorithms/compete/save_algorithm_results",
        output_path=output_path,
        data=log_table.describe(),
    ):
        log_table.to_csv(output_path, index=False, sep="\t")

    return log_table, trained_algorithms


def get_algorithm_name(algorithm):
    """
    Get the name of a provided algorithm.

    Args:
        algorithm: Initialized algorithm.

    :return: algorithm_name *(str)*: \n
        Name of the provided algorithm.
    """

    algorithm_name = algorithm.__class__.__name__

    if algorithm_name in ["AdaBoostRegressor", "AdaBoostClassifier", "BaggingRegressor", "BaggingClassifier"]:
        algorithm_name += f"_{algorithm.estimator.__class__.__name__}"

    return algorithm_name


def get_tuning_hyperparams(module, random_state):
    """
    Get tuning hyperparameters for model tuning for the given module.

    Args:
        module (str): GenoML module being used.
        random_state (int): Random seed for reproducibility.

    :return: dict_hyperparams *(dict)*: \n
        Hyperparameters for each model.
    """

    dict_hyperparams = {
        "AdaBoostClassifier" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"),
            "random_state": Categorical([random_state]),
        },
        "AdaBoostRegressor" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "loss": Categorical(["linear", "square", "exponential"]),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"),
            "random_state": Categorical([random_state]),
        },
        "BaggingClassifier" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "max_samples": Real(0.1, 1),
            "max_features": Real(0.1, 1),
            "warm_start": Categorical([True, False]),
            "bootstrap": Categorical([True, False]),
            'n_jobs': Categorical([-1]),
            "random_state": Categorical([random_state]),
        },
        "BaggingRegressor" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "max_samples": Real(0.1, 1),
            "max_features": Real(0.1, 1),
            "warm_start": Categorical([True, False]),
            "bootstrap": Categorical([True, False]),
            'n_jobs': Categorical([-1]),
            "random_state": Categorical([random_state]),
        },
        "RandomForestClassifier" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "criterion": Categorical(["gini", "entropy", "log_loss"]),
            "max_depth": Integer(1, 10),
            "min_weight_fraction_leaf": Real(0, 0.5),
            "max_features": Real(0.1, 1),
            "warm_start": Categorical([True, False]),
            "random_state": Categorical([random_state]),
        },
        "RandomForestRegressor" : {
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "criterion": Categorical(["squared_error", "absolute_error", "friedman_mse", "poisson"]),
            "max_depth": Integer(1, 10),
            "min_weight_fraction_leaf": Real(0, 0.5),
            "max_features": Real(0.1, 1),
            "warm_start": Categorical([True, False]),
            "random_state": Categorical([random_state]),
        },
        "XGBRegressor" : {
            "max_depth": Integer(1, 10), 
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"), 
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"), 
            "gamma": Real(1e-4, 1e2, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Integer(1, 10),
            "random_state": Categorical([random_state]),
        },
        "GradientBoostingClassifier" : {
            "loss": Categorical(["log_loss", "exponential"]),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"),
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "subsample": Real(0.5, 1),
            "criterion": Categorical(["friedman_mse", "squared_error"]),
            "min_samples_split": Real(0.01, 1),
            "min_samples_leaf": Real(0.01, 0.99),
            "min_weight_fraction_leaf": Real(0, 0.5),
            "max_depth": Integer(1, 10),
            "random_state": Categorical([random_state]),
            "max_features": Categorical(["sqrt", "log2"]),
        },
        "GradientBoostingRegressor" : {
            "loss": Categorical(["squared_error", "absolute_error", "huber", "quantile"]),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"),
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "subsample": Real(0.5, 1),
            "criterion": Categorical(["friedman_mse", "squared_error"]),
            "min_samples_split": Real(0.01, 1),
            "min_samples_leaf": Real(0.01, 0.99),
            "min_weight_fraction_leaf": Real(0, 0.5),
            "max_depth": Integer(1, 10),
            "random_state": Categorical([random_state]),
            "max_features": Categorical(["sqrt", "log2"]),
        },
        "ComplementNB" : {
            "alpha": Real(1e-3, 1e0, prior="log-uniform"),
        },
        "ElasticNet" : {
            "alpha": Real(1e-3, 1e0, prior="log-uniform"),
            "l1_ratio": Real(0, 1),
            "selection": Categorical(["cyclic", "random"]),
            "max_iter": Categorical([5000]),
        },
        "KNeighborsClassifier" : {
            "leaf_size": Integer(1e0, 1e2, prior="log-uniform"),
            "n_neighbors": Integer(1, 10),
            "weights": Categorical(["uniform", "distance"]),
            "algorithm": Categorical(["ball_tree", "kd_tree", "brute"]),
            "p": Integer(1, 2),
        },
        "KNeighborsRegressor" : {
            "leaf_size": Integer(1e0, 1e2, prior="log-uniform"), 
            "n_neighbors": Integer(1, 10),
            "weights": Categorical(["uniform", "distance"]),
            "algorithm": Categorical(["ball_tree", "kd_tree", "brute"]),
            "p": Integer(1, 2),
        },
        "LinearDiscriminantAnalysis" : [
            {
                "solver": Categorical(["svd"]),
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),
            },
            {
                "solver": Categorical(["lsqr", "eigen"]),
                "shrinkage": Real(1e-4, 1.0, prior="log-uniform"),
                "tol": Real(1e-6, 1e-2, prior="log-uniform"),
            }
        ],
        # "MLPClassifier" : {
        #     "hidden_layer_sizes": Categorical([50, 100, 200]),
        #     "activation": Categorical(["logistic", "tanh", "relu"]),
        #     "solver": Categorical(["sgd", "adam"]),
        #     "alpha": Real(1e-5, 1e0, prior="log-uniform"),
        #     "learning_rate": Categorical(['constant', 'invscaling', 'adaptive']),
        #     "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        #     "max_iter": Categorical([500]),
        # },
        # "MLPRegressor" : {
        #     "hidden_layer_sizes": Categorical([50, 100, 200]),
        #     "activation": Categorical(["logistic", "tanh", "relu"]),
        #     "solver": Categorical(["sgd", "adam"]),
        #     "alpha": Real(1e-5, 1e0, prior="log-uniform"), 
        #     "learning_rate": Categorical(['constant', 'invscaling', 'adaptive']),
        #     "learning_rate_init": Real(1e-4, 1e-2, prior="log-uniform"),
        #     "max_iter": Categorical([500]),
        # },
        "QuadraticDiscriminantAnalysis" : {
            "tol": Real(1e-6, 1e-2, prior="log-uniform"),
        },
        "SGDClassifier" : {
            "loss": Categorical(["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"]),
            "penalty": Categorical(["elasticnet"]),
            "alpha": Real(1e-5, 1e0, prior="log-uniform"),
            "l1_ratio": Real(0, 1),
            "n_jobs": Categorical([-1]),
            "learning_rate": Categorical(["constant", "optimal", "invscaling", "adaptive"]),
        },
        "SGDRegressor" : {
            "loss": Categorical(["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"]),
            "penalty": Categorical(["elasticnet"]),
            "alpha": Real(1e-5, 1e0, prior="log-uniform"), 
            "l1_ratio": Real(0, 1),
            "learning_rate": Categorical(["constant", "optimal", "invscaling", "adaptive"]),
        },
        "SVC" : {
            "C": Integer(1, 10),
            "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]),
            "degree": Integer(1, 4),
            "gamma": Categorical(["scale", "auto"]),
            "coef0": Real(0, 1),
            "probability": Categorical([True]),
        },
        "SVR" : {
            "kernel": Categorical(["linear", "poly", "rbf", "sigmoid"]), 
            "degree": Integer(1, 4),
            "gamma": Categorical(["scale", "auto"]),
            "coef0": Real(0, 1),
            "C": Integer(1, 10),
        },
    }
    
    if module == "discrete":
        dict_hyperparams["LogisticRegression"] = {
            "penalty": Categorical(["l1", "l2"]),
            "C": Integer(1e0, 1e1),
            "solver": Categorical(["liblinear", "saga"]),
            "random_state": Categorical([random_state]),
        }
        dict_hyperparams["XGBClassifier"] = {
            "max_depth": Integer(1, 10),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"), 
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "gamma": Real(1e-4, 1e2, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Integer(1, 10),
            "objective": Categorical(["binary:logistic"]),
            "random_state": Categorical([random_state]),
        }

    elif module == "multiclass":
        dict_hyperparams["LogisticRegression"] = {
            "penalty": Categorical(["l1", "l2"]),
            "C": Integer(1e0, 1e1),
            "solver": Categorical(["saga"]),
            "random_state": Categorical([random_state]),
        }
        dict_hyperparams["XGBClassifier"] = {
            "max_depth": Integer(1, 10),
            "learning_rate": Real(1e-4, 1e0, prior="log-uniform"), 
            "n_estimators": Integer(1e1, 1e2, prior="log-uniform"),
            "gamma": Real(1e-4, 1e2, prior="log-uniform"),
            "subsample": Real(0.5, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Integer(1, 10),
            "objective": Categorical(["multi:softprob"]),
            "random_state": Categorical([random_state]),
        }
    
    return dict_hyperparams


def _fit_algorithm(algorithm, algorithm_name, x_train, y_train, x_valid, y_valid, column_names, calculate_accuracy_scores):
    with Timer() as timer:
        algorithm = copy.deepcopy(algorithm)
        algorithm.fit(x_train, y_train)

    row = [algorithm_name, timer.elapsed()] + list(calculate_accuracy_scores(x_valid, y_valid, algorithm))

    results_str = metrics_to_str(dict(zip(column_names, row)))
    DescriptionLoader.print(
        "utils/training/fit_algorithms/compete/algorithm/results",
        name=algorithm_name, 
        results=results_str,
    )
        
    return row, algorithm


def train_valid_split(df, train_split, random_state):
    if train_split > 1:
        train_split = train_split / 100

    y = df.PHENO
    x = df.drop(columns=['PHENO'])
    x_train, x_valid, y_train, y_valid = model_selection.train_test_split(
        x, 
        y, 
        test_size=1-train_split, 
        random_state=random_state,
    )

    return x_train, x_valid, y_train, y_valid
