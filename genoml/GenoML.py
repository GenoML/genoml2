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

import argparse
import functools

from genoml.cli import continuous_supervised_train
from genoml.cli import continuous_supervised_tune
from genoml.cli import discrete_supervised_train
from genoml.cli import discrete_supervised_tune
from genoml import utils


def main():
    parser = argparse.ArgumentParser()
    # These are mandatory 
    parser.add_argument("data", choices=["discrete", "continuous"])
    parser.add_argument("method", choices=["supervised", "unsupervised"])
    parser.add_argument("mode", choices=["train", "tune"])

    # Global
    parser.add_argument("--prefix", type=str, default="GenoML_data", help="Prefix for your training data build.")
    parser.add_argument('--metric_max', type=str, default='AUC',
                        choices=['AUC', "Balanced_Accuracy", "Specificity", "Sensitivity"],
                        help='How do you want to determine which algorithm performed the best? [default: AUC].')
    parser.add_argument('--verbose', action='store_true', default=False, help="Verbose output.")

    # TRAINING

    # Discrete
    parser.add_argument('--prob_hist', type=bool, default=False)
    parser.add_argument('--auc', type=bool, default=False)

    # Continuous
    parser.add_argument('--export_predictions', type=bool, default=False)

    # TUNING
    parser.add_argument('--metric_tune', type=str, default='AUC', choices=['AUC', "Balanced_Accuracy"],
                        help='Using what metric of the best algorithm do you want to tune on? [default: AUC].')
    parser.add_argument('--max_tune', type=int, default=50,
                        help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
    parser.add_argument('--n_cv', type=int, default=5,
                        help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')

    args = parser.parse_args()
    utils.ContextScope._verbose = args.verbose

    entry = {
        ("discrete", "supervised", "train"): (discrete_supervised_train.main,
                                              (args.prefix, args.metric_max, args.prob_hist, args.auc)),
        ("discrete", "supervised", "tune"): (discrete_supervised_tune.main,
                                             (args.prefix, args.metric_tune, args.max_tune, args.n_cv)),
        ("continuous", "supervised", "train"): (continuous_supervised_train.main,
                                                (args.prefix, args.export_predictions)),
        ("continuous", "supervised", "tune"): (continuous_supervised_tune.main,
                                               (args.prefix, args.max_tune, args.n_cv))
    }[(args.data, args.method, args.mode)]
    entry[0](*entry[1])


if __name__ == "__main__":
    main()
