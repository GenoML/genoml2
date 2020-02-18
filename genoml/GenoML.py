# Import the necessary packages 
import argparse
from functools import partial

# Import the necessary internal GenoML packages 
from genoml.cli import dstrain
from genoml.cli import dstune
from genoml.cli import cstrain
from genoml.cli import cstune
   
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", choices=["discrete", "continuous"])
    parser.add_argument("method", choices=["supervised", "unsupervised"])
    parser.add_argument("mode", choices=["train", "tune"])

    #Global
    parser.add_argument("--prefix", type=str, default="GenoML_data", help="Prefix for your training data build.")


    # TRAINING 
    parser.add_argument('--rank_features', type=str, default='skip', choices=['skip','run'], help='Export feature rankings: (skip, run). Exports feature rankings but can be quite slow with huge numbers of features [default: skip].')
        
        # Discrete 
    parser.add_argument('--prob_hist', type=bool, default=False)
    parser.add_argument('--auc', type=bool, default=False)

        # Continuos 
    parser.add_argument('--export_predictions', type=bool, default=False)

    # TUNING
    parser.add_argument('--max_tune', type=int, default=50, help='Max number of tuning iterations: (integer likely greater than 10). This governs the length of tuning process, run speed and the maximum number of possible combinations of tuning parameters [default: 50].')
    parser.add_argument('--n_cv', type=int, default=5, help='Number of cross validations: (integer likely greater than 3). Here we set the number of cross-validation runs for the algorithms [default: 5].')

    args = parser.parse_args()

    # DICTIONARY OF CLI 
    clis = {
    "discretesupervisedtrain": partial(dstrain, args.prefix, args.rank_features, args.prob_hist, args.auc),
    "discretesupervisedtune": partial(dstune, args.prefix, args.max_tune, args.n_cv),
    "continuoussupervisedtrain": partial(cstrain, args.prefix, args.rank_features, args.export_predictions),
    "continuoussupervisedtune": partial(cstune, args.prefix, args.max_tune, args.n_cv)
    }

    # Process the arguments 
    clis[args.data + args.method + args.mode]()


    
