#! /usr/bin/env python -u
# coding=utf-8
import glob
import json
import os
import shutil
import sys
import tempfile

from genoml.check_dependencies import check_dependencies
from genoml.parse_arguments import Options
from genoml.steps.data_prune import DataPruneStep
from genoml.steps.model_train import ModelTrainStep
from genoml.steps.model_tune import ModelTuneStep
from genoml.steps.model_validate import ModelValidateStep

sys.tracebacklimit = 0

def cli():
    options = Options("commandline_args.txt")
    dependencies = check_dependencies()

    if options.is_data_prune():
        process = DataPruneStep()
    elif options.is_model_train():
        process = ModelTrainStep()
    elif options.is_model_tune():
        process = ModelTuneStep()
    elif options.is_model_validate():
        process = ModelValidateStep()
    else:
        raise Exception("ISSUE: received unrecognizable option.")

    process.set_environment(options, dependencies)
    process.process()


def train():
    options = Options("train_commandline_args.txt")
    dependencies = check_dependencies()
    tmp_dir = tempfile.mkdtemp()
    options._options['--prune-prefix'] = os.path.join(tmp_dir, "model")
    options._options['--best-model-name'] = "best_model"
    print(tmp_dir)

    for process in [DataPruneStep(), ModelTrainStep(), ModelTuneStep()]:
        process.set_environment(options, dependencies)
        process.process()

    model_name = os.path.join(tmp_dir, "model.genoml_model")
    copy_originals(options, tmp_dir)
    shutil.make_archive(model_name, 'bztar', os.path.dirname(options.prefix))
    shutil.move(model_name + ".tar.bz2", options.model_file)
    shutil.rmtree(tmp_dir, True)


def inference():
    options = Options("inference_commandline_args.txt")
    dependencies = check_dependencies()

    os.makedirs(options.valid_dir)
    tmp_dir = tempfile.mkdtemp()
    print(tmp_dir)
    shutil.unpack_archive(options.model_file, tmp_dir, format="bztar")
    load_originals(options, tmp_dir)
    options._options['--prune-prefix'] = os.path.join(tmp_dir, "model")
    print(options._options)

    for process in [ModelValidateStep()]:
        process.set_environment(options, dependencies)
        process.process()

    for file in glob.glob(tmp_dir + "/model_validation*"):
        shutil.copy(file, options.valid_dir)
    shutil.rmtree(tmp_dir, True)


def copy_originals(options, base_name):
    names = {}
    for group in ["--pheno-file", "--geno-prefix", "--gwas-file"]:
        if group not in options._options or options._options[group] is None or options._options[group] == "":
            continue
        name = group[2:]
        target = os.path.join(base_name, name)
        os.makedirs(target)
        for file in glob.glob(options._options[group] + "*"):
            shutil.copy(file, target)
        names[group] = os.path.basename(options._options[group])
    names["--impute-data"] = options.impute_data
    with open(os.path.join(base_name, "names.json"), "w") as fp:
        json.dump(names, fp)


def load_originals(options, base_name):
    with open(os.path.join(base_name, "names.json")) as fp:
        names = json.load(fp)

    for group, value in names.items():
        if group == "--impute-data":
            options._options["--impute-data"] = value
            continue
        name = group[2:]
        target = os.path.join(base_name, name)
        options._options[group] = os.path.join(target, value)


if __name__ == '__main__':
    cli()
