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


from sklearn import discriminant_analysis, ensemble, linear_model, neighbors, neural_network, svm
import xgboost


### TODO: Look into different estimators for AdaBoost/Bagging?
### TODO: Weird results for: SGDClassifier, QuadraticDiscriminantAnalysis
CANDIDATE_ALGORITHMS = {
    "discrete_supervised": [
        discriminant_analysis.LinearDiscriminantAnalysis(),
        discriminant_analysis.QuadraticDiscriminantAnalysis(),
        ensemble.AdaBoostClassifier(random_state=3),
        ensemble.BaggingClassifier(random_state=3),
        ensemble.GradientBoostingClassifier(random_state=3),
        ensemble.RandomForestClassifier(n_estimators=100, random_state=3),
        linear_model.LogisticRegression(solver='lbfgs', random_state=3),
        linear_model.SGDClassifier(loss='modified_huber', random_state=3),
        neighbors.KNeighborsClassifier(),
        neural_network.MLPClassifier(random_state=3),
        svm.SVC(probability=True, gamma='scale', random_state=3),
        xgboost.XGBClassifier(random_state=3),
    ],
    "continuous_supervised": [
        ensemble.AdaBoostRegressor(random_state=3),
        ensemble.BaggingRegressor(random_state=3),
        ensemble.GradientBoostingRegressor(random_state=3),
        ensemble.RandomForestRegressor(random_state=3),
        linear_model.ElasticNet(random_state=3),
        linear_model.SGDRegressor(random_state=3),
        neighbors.KNeighborsRegressor(),
        neural_network.MLPRegressor(random_state=3),
        svm.SVR(gamma='auto'),
        xgboost.XGBRegressor(random_state=3),
    ],
}


def get_candidate_algorithms(module_name):
    return CANDIDATE_ALGORITHMS.get(module_name, {})
