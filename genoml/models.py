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


import xgboost
from sklearn import discriminant_analysis, ensemble, linear_model, neighbors, neural_network, svm


CANDIDATE_ALGORITHMS = {
    "discrete_supervised": [
        lambda random_state: discriminant_analysis.LinearDiscriminantAnalysis(),
        # lambda random_state: discriminant_analysis.QuadraticDiscriminantAnalysis(),
        lambda random_state: ensemble.AdaBoostClassifier(random_state=random_state),
        lambda random_state: ensemble.BaggingClassifier(random_state=random_state),
        lambda random_state: ensemble.GradientBoostingClassifier(random_state=random_state),
        lambda random_state: ensemble.RandomForestClassifier(n_estimators=100, random_state=random_state),
        lambda random_state: linear_model.LogisticRegression(solver='lbfgs', random_state=random_state),
        lambda random_state: linear_model.SGDClassifier(loss='modified_huber', random_state=random_state),
        lambda random_state: neighbors.KNeighborsClassifier(),
        # lambda random_state: neural_network.MLPClassifier(random_state=random_state),
        lambda random_state: svm.SVC(probability=True, gamma='scale', random_state=random_state),
        lambda random_state: xgboost.XGBClassifier(random_state=random_state),
    ],
    "continuous_supervised": [
        lambda random_state: ensemble.AdaBoostRegressor(random_state=random_state),
        lambda random_state: ensemble.BaggingRegressor(random_state=random_state),
        lambda random_state: ensemble.GradientBoostingRegressor(random_state=random_state),
        lambda random_state: ensemble.RandomForestRegressor(random_state=random_state),
        lambda random_state: linear_model.ElasticNet(random_state=random_state),
        lambda random_state: linear_model.SGDRegressor(random_state=random_state),
        lambda random_state: neighbors.KNeighborsRegressor(),
        # lambda random_state: neural_network.MLPRegressor(random_state=random_state),
        lambda random_state: svm.SVR(gamma='auto'),
        lambda random_state: xgboost.XGBRegressor(random_state=random_state),
    ],
}


def get_candidate_algorithms(module_name, random_state):
    return [constructor(random_state) for constructor in CANDIDATE_ALGORITHMS.get(module_name, [])]
