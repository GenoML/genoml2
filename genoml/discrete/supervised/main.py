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

import sys
from genoml.discrete.supervised.training import Train
from genoml.discrete.supervised.tuning import Tune
from genoml.discrete.supervised.testing import Test


def train(prefix, metric_max):
    trainer = Train(prefix, metric_max)
    trainer.compete()
    trainer.select_best_algorithm()
    trainer.export_model()
    trainer.plot_results()
    trainer.export_prediction_data()


### TODO: Add variables for loading old results
def tune(prefix, metric_tune, max_iter, cv_count):
    tuner = Tune(prefix, metric_tune, max_iter, cv_count)
    tuner.tune_model()
    tuner.report_tune()
    tuner.summarize_tune()
    tuner.compare_performance()
    tuner.plot_results()
    tuner.export_prediction_data()


def test(prefix):
    tester = Test(prefix)
    tester.plot_results()
    tester.export_prediction_data()
    tester.additional_sumstats()
