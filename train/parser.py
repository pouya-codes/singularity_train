import argparse
from submodule_utils.manifest.arguments import manifest_arguments

from submodule_utils import (BALANCE_PATCHES_OPTIONS, DATASET_ORIGINS,
        PATCH_PATTERN_WORDS)
from submodule_utils.arguments import (
        dir_path, file_path, dataset_origin, balance_patches_options,
        str_kv, int_kv, subtype_kv, make_dict,
        ParseKVToDictAction, CustomHelpFormatter)

from train import *

description="""Trains a model for patch classification. This process does the training in the following manner:

 (1) Takes in a JSON file (aka. file of one or more chunks) that is either a split JSON file created by `docker_create_cross_validation_groups`, or a group JSON file created by `docker_create_groups` specified by --chunk_file_location. Each chunk contains patch paths to feed into the classifier. Use --training_chunks to select the chunks to include in your training set, etc. JSON files use Mitch's format for groups i.e. it is a json file with the format

{
    "chunks": [
        {
            "id": int,
            "imgs": list of paths to patches
        },
        ...
    ]
}

 (2) The flag --model_config_location specifies a path to a JSON file containing model hyperparameters. It is a residue of an old config format that didn't have a time to get refactored. For most of the experiments at AIM Lab, we currently use the below config JSON. The primary change is to set the num_subtypes

{
    "num_subtypes" : <number>,
    "deep_model" : "vgg19_bn",
    "use_weighted_loss" : false,
    "continue_train" : false,
    "parameters" : {
        "pretrained" : true
    },
    "optimizer" : {
        "type" : "Adam",
        "parameters" : {
            "lr" : 0.0002,
            "amsgrad" : true,
            "weight_decay" : 0.0005
        }
    }
}

 (3) For each epoch (specified by --epochs), we train the classifier using all patches in the training set, feeding the classifier a batch of patches (with size specified by --batch_size). At every batch interval (specififed by --validation_interval) we run validation loop and save (or overwrite) the model if it achieves the as of yet highest validation accuracy.

 (4) All argument flags used in `docker_train` are saved to a YAML section in the log file (specified by --log_dir_location). Components using the model like `docker_evaluate` reads this log file to find the saved model and model parameters."""

epilog="""
See JIRA tickets for updating docker_{train, evaluate}
"""

@manifest_arguments(default_component_id="try_me", description=description, epilog=epilog)
def create_parser(parser):

        parser.add_argument("--experiment_name", type=str, required=True,
                help="Experiment name used to name log, model outputs.")

        parser.add_argument("--batch_size", type=int, required=True,
                help="Batch size is the number of patches to put in a batch. "
                "This flag sets the batch size to use on training, validation and test datasets.")

        parser.add_argument("--validation_interval", type=int, required=True,
                help="The interval of the training loop to start validating model.")

        parser.add_argument("--epochs", type=int, required=True,
                help="The number of epochs to run model training on training dataset.")

        parser.add_argument("--training_chunks", nargs="+", type=int,
                default=[0],
                help="Space separated number IDs specifying chunks to use for training.")

        parser.add_argument("--validation_chunks", nargs="+", type=int,
                default=[1],
                help="Space separated number IDs specifying chunks to use for validation.")

        parser.add_argument("--is_binary", action='store_true',
                help="Whether we want to categorize patches by the Tumor/Normal category (true) "
                "or by the subtype category (false).")

        parser.add_argument("--subtypes", nargs='+', type=subtype_kv,
                action=ParseKVToDictAction, default=default_subtypes,
                help="space separated words describing subtype=groupping pairs for this study. "
                "Example: if doing one-vs-rest on the subtypes MMRD vs P53ABN, P53WT and POLE "
                "then the input should be 'MMRD=0 P53ABN=1 P53WT=1 POLE=1'")

        parser.add_argument("--patch_pattern", type=str,
                default=default_patch_pattern,
                help="'/' separated words describing the directory structure of the "
                f"patch paths. The words are {tuple(PATCH_PATTERN_WORDS)}. "
                "A non-multiscale patch can be contained in a directory "
                "/path/to/patch/rootdir/Tumor/MMRD/VOA-1234/1_2.png so its patch_pattern is "
                "annotation/subtype/slide. A multiscale patch can be contained in a "
                "directory /path/to/patch/rootdir/Stroma/P53ABN/VOA-1234/10/3_400.png so "
                "its patch pattern is annotation/subtype/slide/magnification")

        parser.add_argument("--chunk_file_location", type=file_path, required=True,
                help="File path of group or split file (aka. chunks) to use "
                "(i.e. /path/to/patient_3_groups.json)")

        parser.add_argument("--log_dir_location", type=dir_path, required=True,
                help="Path to log directory to save training logs (i.e. "
                "/path/to/logs/training/).")

        parser.add_argument("--model_dir_location", type=dir_path, required=True,
                help="Path to model directory to save trained model (i.e. /path/to/model/).")

        parser.add_argument("--model_config_location", type=str, required=True,
                help="Path to model config JSON (i.e. /path/to/model_config.json).")

        parser.add_argument("--patch_location", type=dir_path, required=False,
                help="Path to root directory containing dataset patches specified in "
                "group or split file (i.e. /path/to/patch/rootdir/). Used by Docker "
                "to link the directory.")

        parser.add_argument("--num_patch_workers", type=int,
                default=default_num_patch_workers,
                help="Number of loader worker processes to multi-process data loading. "
                "Default uses single-process data loading.")

        parser.add_argument("--num_validation_batches", type=int, required=False,
                help="Number of validation batches to use for model validation. "
                "Default uses all validation batches for each validation loop.")

        parser.add_argument("--gpu_id", type=int, required=False,
                help="The ID of GPU to select. Default uses GPU with the most free memory.")

        parser.add_argument("--seed", type=int,
                default=default_seed,
                help="Seed for random shuffle.")

        parser.add_argument("--training_shuffle", action='store_true',
                help="Shuffle the training set.")

        parser.add_argument("--validation_shuffle", action='store_true',
                help="Shuffle the validation set.")

        parser.add_argument("--writer_log_dir_location", type=str, required=False,
                help="Directory in log_dir_location to put TensorBoard logs."
                "Default uses log_dir_location/experiment_name.")



        subparsers = parser.add_subparsers(dest="subparser", required=False)

        parser_early_stopping = subparsers.add_parser("early_stopping")

        parser_early_stopping.add_argument("--use_early_stopping", type=bool, default=False, required=False,
                help="Weather or not use EarlyStopping"
                "Default uses False")
        parser_early_stopping.add_argument("--patience", type=int, default=7, required=False,
                help="How long to wait after last time validation loss improved."
                     "Default: 7")
        parser_early_stopping.add_argument("--delta", type=float, default=0, required=False,
                help="Minimum change in the monitored quantity to qualify as an improvement."
                "Default: 0")

        # parser_test_model = subparsers.add_parser("test_model")

        parser_early_stopping.add_argument("--testing_model", type=bool, default=True, required=True,
                help="Weather or not test the model performance after training"
                "Default uses False")

        parser_early_stopping.add_argument("--test_log_dir_location", type=str, required=True,
                help="Location of results of the testing")

        parser_early_stopping.add_argument("--detailed_test_result", type=bool, default=False, required=False,
                help="Weather or not provides deatailed test results, including paths to image files, predicted label, target label, probabilities of classes"
                "Default uses False")

        parser_early_stopping.add_argument("--testing_shuffle", type=bool, default=False,
                            help="Shuffle the testing set.")

        parser_early_stopping.add_argument("--test_chunks", nargs="+", type=int,
                default=[2],
                help="Space separated number IDs specifying chunks to use for testing.")


def get_args():
        parser = create_parser()
        args = parser.get_args()
        return args

