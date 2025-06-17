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

import json
import os
import time
import traceback
import sys
from sklearn import model_selection
from pathlib import Path
import pandas as pd
import joblib
from scipy import stats

__author__ = 'Sayed Hadi Hashemi'

import textwrap


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
    best_algorithm_name = log_table.iloc[best_id].Algorithm
    best_algorithm = algorithms[best_algorithm_name]
    best_algorithm_metrics = log_table.iloc[best_id].to_dict()

    DescriptionLoader.print(
        "utils/training/compete/algorithm/best",
        algorithm=best_algorithm_name,
        metrics=metrics_to_str(best_algorithm_metrics),
    )

    return best_algorithm


def tune_model(estimator, x, y, param_distributions, scoring, n_iter, cv):
    """
    Apply randomized search to fine-tune the selected model.

    Args:
        estimator: Trained baseline model.
        x (pandas.DataFrame): Model input features.
        y (pandas.DataFrame): Reported output features.
        param_distributions (dict): Hyperparameters and corresponsing values to be tested.
        scoring (sklearn.metrics._scorer._Scorer): Scoring metric to evaluate accuracy.
        n_iter (int): Maximum number of iterations.
        cv (int): Number of cross-validations.
    
    :return: cv_results *(dict)*: \n
        Results from hyperparameter tuning.
    :return: algo_tuned: \n
        Tuned model.
    """

    rand_search = model_selection.RandomizedSearchCV(
        estimator = estimator,
        param_distributions = param_distributions,
        scoring = scoring,
        n_iter = n_iter,
        cv = cv,
        n_jobs = -1,
        random_state = 3,
        verbose = 0,
    )

    with Timer() as timer:
        rand_search.fit(x, y)
    print(f"RandomizedSearchCV took {timer.elapsed():.2f} seconds for {n_iter:d} "
          "candidates parameter iterations.")

    cv_results = rand_search.cv_results_
    algo_tuned = rand_search.best_estimator_
    return cv_results, algo_tuned


def sumarize_tune(out_dir, estimator_baseline, estimator_tune, x, y, scoring, cv):
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
        cv (int): Number of cross validations.

    :return: cv_baseline *(pandas.DataFrame)*: \n
        Cross-validation results for the trained baseline model.
    :return: cv_tuned *(pandas.DataFrame)*: \n
        Cross-validation results for the tuned model.
    """

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
    log_outfile = out_dir.joinpath('cv_summary.txt')
    df_cv_summary.to_csv(log_outfile, sep="\t")

    print("Here is the cross-validation summary of your best tuned model hyperparameters...")
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


def report_best_tuning(out_dir, cv_results, n_top):
    """
    Find the top-performing tuning iterations and save those to a table.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_results (dict): Results from hyperparameter tuning.
        n_top (int): Number of iterations to report.
    """

    print("Here is a summary of the top 10 iterations of the hyperparameter tuning...")
    cv_results = pd.DataFrame(cv_results)
    cv_results.sort_values(by='rank_test_score', ascending=True, inplace=True)
    cv_results = cv_results.iloc[:n_top,:]
    for i in range(len(cv_results)):
        current_iteration = cv_results.iloc[i,:]
        print(f"Model with Rank {i + 1}:")
        print(f"Mean Validation Score: {current_iteration['mean_test_score']:.3f} (std: {current_iteration['std_test_score']:.3f})")
        print(f"Parameters: {current_iteration['params']}")
        print("")
    log_outfile = out_dir.joinpath('tuning_summary.txt')
    cv_results.to_csv(log_outfile, index=False, sep="\t")
    print(f"We are exporting a summary table of the top {n_top} iterations of the hyperparameter tuning step and its parameters here {log_outfile}.")


def compare_tuning_performance(out_dir, cv_tuned, cv_baseline, algo_tuned, algo_baseline, x=None):
    """
    Determine whether the fine-tuned model outperformed the baseline model.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        cv_tuned (pandas.DataFrame): Cross-validation results for the tuned model.
        cv_baseline (pandas.DataFrame): Cross-validation results for the trained baseline model.
        algo_tuned: Tuned model.
        algo_baseline: Trained baseline model.
        x (pandas.DataFrame, optional): Model input features (Default: None).
    
    :return: algorithm: \n
        Better-perorming of the fine-tuned and trained baseline models.
    :return: y_predicted *(numpy.ndarray)*: \n
        Predicted outputs from the chosen model.
    """

    print("")

    if cv_baseline.mean() >= cv_tuned.mean():
        print("Based on comparisons of the default parameters to your hyperparameter tuned model, the baseline model actually performed better.")
        print("Looks like the tune wasn't worth it, we suggest either extending the tune time or just using the baseline model for maximum performance.")
        print("")
        print("Let's shut everything down, thanks for trying to tune your model with GenoML.")
        algorithm = algo_baseline
        yield algorithm

    if cv_baseline.mean() < cv_tuned.mean():
        print("Based on comparisons of the default parameters to your hyperparameter tuned model, the tuned model actually performed better.")
        print("Looks like the tune was worth it, we suggest using this model for maximum performance, lets summarize and export this now.")
        print("In most cases, if opting to use the tuned model, a separate test dataset is a good idea. GenoML has a module to fit models to external data.")
        algorithm = algo_tuned
        yield algorithm

    export_model(out_dir.parent, algorithm)

    if x is not None:
        y_predicted = algorithm.predict(x)
        yield y_predicted


def read_munged_data(out_dir, dataset_type):
    """
    Read munged hdf5 file to pandas

    Args:
        out_dir (pathlib.Path): Path to output directory.
    
    :return: df: \n
        Munged dataset.
    """

    infile_h5 = Path(out_dir).joinpath("Munge").joinpath(f"{dataset_type}_dataset.h5")
    with DescriptionLoader.context(
        "read_munge", 
        path=infile_h5,
    ):
        df = pd.read_hdf(infile_h5, key="dataForML")

    DescriptionLoader.print(
        "data_summary", 
        data=df.describe(),
    )

    return df


def export_model(out_dir, algorithm):
    """
    Export a fitted algorithm to a readable file.

    Args:
        out_dir (pathlib.Path): Path to output directory.
        algorithm: Fitted algorithm being exported.
    """

    output_path = out_dir.joinpath('model.joblib')
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

    for algorithm_name, algorithm in algorithms.items():
        with DescriptionLoader.context(
            "utils/training/fit_algorithms/compete/algorithm",
            name=algorithm_name,
        ):
            with Timer() as timer:
                algorithm.fit(x_train, y_train)

            row = [algorithm_name, timer.elapsed()] + calculate_accuracy_scores(x_valid, y_valid, algorithm)

            results_str = metrics_to_str(dict(zip(column_names, row)))
            with DescriptionLoader.context(
                "utils/training/fit_algorithms/compete/algorithm/results",
                name=algorithm_name, 
                results=results_str,
            ):
                log_table.append(row)

    log_table = pd.DataFrame(data=log_table, columns=column_names)
    output_path = out_dir.joinpath('withheld_performance_metrics.txt')
    with DescriptionLoader.context(
        "utils/training/fit_algorithms/compete/save_algorithm_results",
        output_path=output_path,
        data=log_table.describe(),
    ):
        log_table.to_csv(output_path, index=False, sep="\t")

    return log_table


def get_tuning_hyperparams(module):
    """
    Get tuning hyperparameters for model tuning for the given module.

    Args:
        module (str): GenoML module being used.

    :return: dict_hyperparams *(dict)*: \n
        Hyperparameters for each model.
    """

    if module == "continuous":
        dict_hyperparams = {
            'AdaBoostRegressor' : {
                "n_estimators": stats.randint(1, 1000),
                "loss": ["linear", "square", "exponential"],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
            },
            'BaggingRegressor' : {
                "n_estimators": stats.randint(1, 1000),
            },
            'GradientBoostingRegressor' : {
                "n_estimators": stats.randint(1, 1000),
                "loss": ["squared_error", "absolute_error", "huber", "quantile"],
                "learning_rate": [0.0001, 0.001, 0.01, 0.1, 1],
                "subsample": stats.uniform(0, 1),
                "criterion": ["friedman_mse", "squared_error"],
                "min_weight_fraction_leaf": stats.uniform(0, 0.5),
                "max_depth": stats.randint(1, 10),
            },
            'RandomForestRegressor' : {
                "n_estimators": stats.randint(1, 1000),
                "criterion": ["squared_error", "absolute_error", "friedman_mse", "poisson"],
                "min_weight_fraction_leaf": stats.uniform(0, 0.5),
            },
            'ElasticNet' : {
                "alpha": stats.uniform(0, 1),
                "l1_ratio": stats.uniform(0, 1),
            },
            'SGDRegressor' : {
                "loss": ["squared_error", "huber", "epsilon_insensitive", "squared_epsilon_insensitive"],
                "penalty": ["elasticnet"],
                "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 
                "l1_ratio": stats.uniform(0, 1),
                "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
            },
            'KNeighborsRegressor' : {
                "leaf_size": stats.randint(1, 100), 
                "n_neighbors": stats.randint(1, 10),
                "weights": ["uniform", "distance"],
                "algorithm": ["ball_tree", "kd_tree", "brute"],
                "p": stats.uniform(1, 4),
            },
            'MLPRegressor' : {
                "activation": ["identity", "logistic", "tanh", "relu"],
                "solver": ["lbfgs", "sgd", "adam"],
                "alpha": stats.uniform(0, 1), 
                "learning_rate": ['constant', 'invscaling', 'adaptive'],
                "max_iter": [1000],
            },
            'SVR' : {
                "kernel": ["linear", "poly", "rbf", "sigmoid"], 
                "gamma": ["scale", "auto"],
                "C": stats.randint(1, 10),
            },
            'XGBRegressor' : {
                "max_depth": stats.randint(1, 100), 
                "learning_rate": stats.uniform(0, 1), 
                "n_estimators": stats.randint(1, 100), 
                "gamma": stats.uniform(0, 1),
            },
        }
    
    elif module == "discrete":
        dict_hyperparams = {
            'LogisticRegression' : {
                "penalty": ["l1", "l2"],
                "C": stats.randint(1, 10),
            },
            'SGDClassifier' : {
                'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'loss': ['log_loss'],
                'penalty': ['l2'],
                'n_jobs': [-1],
            },
            'RandomForestClassifier' : {
                "n_estimators": stats.randint(1, 1000),
            },
            'AdaBoostClassifier' : {
                "n_estimators": stats.randint(1, 1000),
            },
            'GradientBoostingClassifier' : {
                "n_estimators": stats.randint(1, 1000),
            },
            'BaggingClassifier' : {
                "n_estimators": stats.randint(1, 1000),
            },
            'SVC' : {
                "kernel": ["linear", "poly", "rbf", "sigmoid"],
                "C": stats.randint(1, 10),
            },
            'ComplementNB' : {
                "alpha": stats.uniform(0, 1),
            },
            'MLPClassifier' : {
                "alpha": stats.uniform(0, 1),
                "learning_rate": ['constant', 'invscaling', 'adaptive'],
                "max_iter": [1000],
            },
            'XGBClassifier' : {
                "max_depth": stats.randint(1, 100),
                "learning_rate": stats.uniform(0, 1),
                "n_estimators": stats.randint(1, 100),
                "gamma": stats.uniform(0, 1),
            },
            'KNeighborsClassifier' : {
                "leaf_size": stats.randint(1, 100),
                "n_neighbors": stats.randint(1, 10),
            },
            'LinearDiscriminantAnalysis' : {
                "tol": stats.uniform(0, 1),
            },
            'QuadraticDiscriminantAnalysis' : {
                "tol": stats.uniform(0, 1),
            },
        }

    elif module == "multiclass":
        dict_hyperparams = {
            'LogisticRegression' : {
                "estimator__penalty": ["l1", "l2"],
                "estimator__C": stats.randint(1, 10),
            },
            'SGDClassifier' : {
                'estimator__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
                'estimator__loss': ['log_loss'],
                'estimator__penalty': ['l2'],
                'n_jobs': [-1],
            },
            'RandomForestClassifier' : {
                "estimator__n_estimators": stats.randint(1, 1000),
            },
            'AdaBoostClassifier' : {
                "estimator__n_estimators": stats.randint(1, 1000),
            },
            'GradientBoostingClassifier' : {
                "estimator__n_estimators": stats.randint(1, 1000),
            },
            'BaggingClassifier' : {
                "estimator__n_estimators": stats.randint(1, 1000),
            },
            'SVC' : {
                "estimator__kernel": ["linear", "poly", "rbf", "sigmoid"],
                "estimator__C": stats.randint(1, 10),
            },
            'ComplementNB' : {
                "alpha": stats.uniform(0, 1),
            },
            'MLPClassifier' : {
                "estimator__alpha": stats.uniform(0, 1),
                "estimator__learning_rate": ['constant', 'invscaling', 'adaptive'],
                "estimator__max_iter": [1000],
            },
            'XGBClassifier' : {
                "estimator__max_depth": stats.randint(1, 100),
                "estimator__learning_rate": stats.uniform(0, 1),
                "estimator__n_estimators": stats.randint(1, 100),
                "estimator__gamma": stats.uniform(0, 1),
            },
            'KNeighborsClassifier' : {
                "estimator__leaf_size": stats.randint(1, 100),
                "estimator__n_neighbors": stats.randint(1, 10),
            },
            'LinearDiscriminantAnalysis' : {
                "estimator__tol": stats.uniform(0, 1),
            },
            'QuadraticDiscriminantAnalysis' : {
                "estimator__tol": stats.uniform(0, 1),
            },
        }
    
    return dict_hyperparams
