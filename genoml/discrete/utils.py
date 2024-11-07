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


def ROC(save, plot_path, ground_truth, predictions, plot_label="Test Dataset"):
    plt.figure()

    fpr, tpr, _ = metrics.roc_curve(ground_truth, predictions)
    roc_auc = metrics.roc_auc_score(ground_truth, predictions)

    plt.plot(fpr, tpr, color='purple', label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='cyan', linestyle='--', label='Chance (area = %0.2f)' % 0.5)

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title(f'Receiver operating characteristic (ROC) - {plot_label}')
    plt.legend(loc="lower right")
    if save:
        plt.savefig(plot_path, dpi=600)
        print(f"We are also exporting an ROC curve for you here {plot_path} this is a graphical representation of AUC "
              f"in the withheld test data for the best performing algorithm.")


def precision_recall_plot(save, plot_path, ground_truth, predictions, plot_label="Test Dataset"):
    plt.figure()

    precision, recall, _ = metrics.precision_recall_curve(ground_truth, predictions)
    plt.plot(precision, recall, label="Precision-Recall curve")

    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision vs. Recall Curve - {plot_label}")
    plt.legend(loc="lower left")
    if save:
        plt.savefig(plot_path, dpi=600)
        print(f"We are also exporting a Precision-Recall plot for you here {plot_path}. This is a graphical "
              f"representation of the relationship between precision and recall scores in the withheld test data for "
              f"the best performing algorithm.")


def export_prediction_tables(algo, y_vals, x_vals, IDs, output_path):
    predicted_probs = algo.predict_proba(x_vals)
    case_probs = predicted_probs[:,1]
    case_probs_df = pd.DataFrame(case_probs)
    predicted_probs_df = pd.DataFrame(predicted_probs, dtype=float)
    predicted_cases_df = predicted_probs_df.idxmax(axis=1)
    y_vals_df = pd.DataFrame(y_vals)
    IDs_df = pd.DataFrame(IDs)
    prediction_df = pd.concat(
        [
            IDs_df.reset_index(),
            y_vals_df.reset_index(drop=True),
            case_probs_df.reset_index(drop=True),
            predicted_cases_df.reset_index(drop=True),
        ],
        axis=1,
        ignore_index=True,
    )

    prediction_df.columns = ['INDEX', 'ID', "CASE_REPORTED", "CASE_PROBABILITY", "CASE_PREDICTED"]
    prediction_df = prediction_df.drop(columns=['INDEX'])
    prediction_df.to_csv(output_path, index=False)

    print("")
    print(f"Preview of the exported predictions for the withheld test data that has been exported as {output_path} "
          f"these are pretty straight forward.")
    print("They generally include the sample ID, the previously reported case status, the probabilities for each case "
          "from the best performing algorithm, and the predicted label from that algorithm")
    print("")
    print("#" * 70)
    print(prediction_df.head())
    print("#" * 70)

    return prediction_df


def export_prob_hist(to_plot_df, plot_path):
    plt.figure()

    # Using the withheld sample data
    to_plot_df[f'percent_probability'] = to_plot_df[f'CASE_PROBABILITY'] * 100
    to_plot_df[f'Probability (%)'] = to_plot_df[f'percent_probability'].round(decimals=0)
    to_plot_df['Reported Status'] = to_plot_df['CASE_REPORTED']
    to_plot_df['Predicted Status'] = to_plot_df['CASE_PREDICTED']

    to_plot_df.describe()

    # Start plotting
    sns.histplot(
        data=to_plot_df,
        x=f"Probability (%)",
        hue="Predicted Status",
        kde=True,
        alpha=0.2,
        multiple='dodge',
    )
    path = f"{plot_path}.png"
    plt.savefig(path, dpi=300)
    plt.clf()
    print(f"We are also exporting probability density plots to the file {path} this is a plot of the probability "
          f"distributions for each case, stratified by case status in the withheld test samples.")


def get_hyperparameters(best_algo):
    if best_algo == 'LogisticRegression':
        hyperparameters = {
            "penalty": ["l1", "l2"],
            "C": stats.randint(1, 10),
        }

    elif best_algo == 'SGDClassifier':
        hyperparameters = {
            'alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
            'loss': ['log_loss'],
            'penalty': ['l2'],
            'n_jobs': [-1],
        }

    elif best_algo in ('RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'BaggingClassifier'):
        hyperparameters = {
            "n_estimators": stats.randint(1, 1000),
        }

    elif best_algo == 'SVC':
        hyperparameters = {
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "C": stats.randint(1, 10),
        }

    elif best_algo == 'ComplementNB':
        hyperparameters = {
            "alpha": stats.uniform(0, 1),
        }

    elif best_algo == 'MLPClassifier':
        hyperparameters = {
            "alpha": stats.uniform(0, 1),
            "learning_rate": ['constant', 'invscaling', 'adaptive'],
        }

    elif best_algo == 'XGBClassifier':
        hyperparameters = {
            "max_depth": stats.randint(1, 100),
            "learning_rate": stats.uniform(0, 1),
            "n_estimators": stats.randint(1, 100),
            "gamma": stats.uniform(0, 1),
        }

    elif best_algo == 'KNeighborsClassifier':
        hyperparameters = {
            "leaf_size": stats.randint(1, 100),
            "n_neighbors": stats.randint(1, 10),
        }

    elif best_algo in ('LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis'):
        hyperparameters = {
            "tol": stats.uniform(0, 1),
        }

    return hyperparameters


def get_best_algo(best_algo):
    if best_algo == 'LogisticRegression':
        algo = getattr(sklearn.linear_model, best_algo)()

    elif best_algo == 'SGDClassifier':
        algo = getattr(sklearn.linear_model, best_algo)(loss='modified_huber')

    elif best_algo in ('RandomForestClassifier', 'AdaBoostClassifier', 'GradientBoostingClassifier', 'BaggingClassifier'):
        algo = getattr(sklearn.ensemble, best_algo)()

    elif best_algo == 'SVC':
        algo = getattr(sklearn.svm, best_algo)(probability=True)

    elif best_algo == 'ComplementNB':
        algo = getattr(sklearn.naive_bayes, best_algo)()

    elif best_algo == 'MLPClassifier':
        algo = getattr(sklearn.neural_network, best_algo)()

    elif best_algo == 'XGBClassifier':
        algo = getattr(xgboost, best_algo)()

    elif best_algo == 'KNeighborsClassifier':
        algo = getattr(sklearn.neighbors, best_algo)()

    elif best_algo in ('LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis'):
        algo = getattr(sklearn.discriminant_analysis, best_algo)()

    return algo


def summary_stats(algo, y_vals, x_vals):

    log_cols = [
        "Algorithm",
        "AUC_Percent",
        "Accuracy_Percent",
        "Balanced_Accuracy_Percent",
        "Log_Loss",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "Runtime_Seconds",
    ]

    name = algo.__class__.__name__
    print("")
    print("#" * 70)
    print("")
    print(name)
    start_time = time.time()
    algo.fit(x_vals, y_vals)

    rocauc, acc, balacc, ll = calculate_accuracy_scores(
        algo,
        y_vals,
        x_vals,
    )

    end_time = time.time()
    elapsed_time = (end_time - start_time)
    print("Runtime in seconds: {:.4}".format(elapsed_time))

    CM = metrics.confusion_matrix(
        y_vals.values,
        algo.predict(x_vals),
    )
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    log_entry = pd.DataFrame(
        [[
            name,
            rocauc * 100,
            acc * 100,
            balacc * 100,
            ll,
            sensitivity,
            specificity,
            PPV,
            NPV,
            elapsed_time,
        ]],
        columns=log_cols,
    )

    return log_entry


def calculate_accuracy_scores(algo, y_vals, x_vals):

    rocauc = metrics.roc_auc_score(
        y_vals,
        algo.predict(x_vals),
        multi_class="ovr",
    )
    print("AUC: {:.4%}".format(rocauc))

    acc = metrics.accuracy_score(
        y_vals,
        algo.predict(x_vals),
    )
    print("Accuracy: {:.4%}".format(acc))

    balacc = metrics.balanced_accuracy_score(
        y_vals.values,
        algo.predict(x_vals),
    )
    print("Balanced Accuracy: {:.4%}".format(balacc))

    ll = metrics.log_loss(
        y_vals,
        algo.predict(x_vals),
    )
    print("Log Loss: {:.4}".format(ll))
    return rocauc, acc, balacc, ll