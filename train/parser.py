import argparse
from submodule_utils.manifest.arguments import manifest_arguments

from submodule_utils import (BALANCE_PATCHES_OPTIONS, DATASET_ORIGINS,
        PATCH_PATTERN_WORDS, set_random_seed, DEAFULT_SEED)
from submodule_utils.arguments import (
        dir_path, str2bool, file_path, dataset_origin, balance_patches_options,
        str_kv, int_kv, subtype_kv, make_dict,
        ParseKVToDictAction, CustomHelpFormatter)

from train import *

default_num_patch_workers = 0
default_subtypes = {'MMRD':0, 'P53ABN': 1, 'P53WT': 2, 'POLE': 3}
default_patch_pattern = 'annotation/subtype/slide'

description="""Trains a model for patch classification. This process does the training in the following manner:

 (1) Takes in a JSON file (aka. file of one or more chunks) that is either a split JSON file created by `singularity_create_cross_validation_groups`, or a group JSON file created by `singularity_create_groups` specified by --chunk_file_location. Each chunk contains patch paths to feed into the classifier. Use --training_chunks to select the chunks to include in your training set, etc. JSON files use Mitch's format for groups i.e. it is a json file with the format

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
        "normalize" : {
        "normalize" : false,
        "mean" : [ 0.485, 0.456, 0.406 ],
        "std" : [ 0.229, 0.224, 0.225 ]
    },
    "augmentation" : true,
    "parameters" : {
        "pretrained" : true
    },
    "optimizer" : {
        "type" : "Adam",
        "parameters" : {
            "lr" : 0.00001,
            "amsgrad" : true,
            "weight_decay" : 0.0005
        }
    }
}


 (3) For each epoch (specified by --epochs), we train the classifier using all patches in the training set, feeding the classifier a batch of patches (with size specified by --batch_size). At every batch interval (specififed by --validation_interval) we run validation loop and save (or overwrite) the model if it achieves the as of yet highest validation accuracy.
"""

epilog="""
"""

@manifest_arguments(default_component_id="try_me", description=description, epilog=epilog)
def create_parser(parser):

        parser.add_argument("--experiment_name", type=str, required=True,
                help="Experiment name used to name log, model outputs.")

        parser.add_argument("--train_model", type=str2bool, nargs='?',
                        const=True, default=True, required=False,
                help="Train the model or just test the model"
                "Default uses False")

        parser.add_argument("--batch_size", type=int, required=True,
                help="Batch size is the number of patches to put in a batch. "
                "This flag sets the batch size to use on training, validation and test datasets.")

        parser.add_argument("--weighted_sampler", type=str2bool, nargs='?',
                        const=True, default=False, required=False,
                help="Whether we use weighted sampler or not."
                     "By using the weighted sampler you can ensure that each batch "
                     "sees a proportional number of all classes")

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

        parser.add_argument("--save_model_for_export", action='store_true',
                help="Whether we want to save the entire model for export")

        parser.add_argument("--model_config_location", type=str, required=True,
                help="Path to model config JSON (i.e. /path/to/model_config.json).")

        parser.add_argument("--num_patch_workers", type=int,
                default=default_num_patch_workers,
                help="Number of loader worker processes to multi-process data loading. "
                "Default uses single-process data loading.")

        parser.add_argument("--num_validation_batches", type=int, required=False,
                help="Number of validation batches to use for model validation. "
                "Default uses all validation batches for each validation loop.")

        parser.add_argument("--gpu_id", type=int, required=False,
                help="The ID of GPU to select. Default uses GPU with the most free memory.")

        parser.add_argument("--number_of_gpus", type=int, required=False, default=1,
                help="The number of GPUs to use. Default uses a GPU with the most free memory.")


        parser.add_argument("--seed", type=int,
                default=DEAFULT_SEED,
                help="Seed for random shuffle.")

        parser.add_argument("--training_shuffle", action='store_true',
                help="Shuffle the training set.")

        parser.add_argument("--validation_shuffle", action='store_true',
                help="Shuffle the validation set.")

        parser.add_argument("--writer_log_dir_location", type=dir_path, required=False,
                help="Directory in log_dir_location to put TensorBoard logs."
                "Default uses log_dir_location/experiment_name.")



        subparsers = parser.add_subparsers(dest="subparser", required=False)

        subparser_parameters = subparsers.add_parser("early_stopping")

        subparser_parameters.add_argument("--use_early_stopping", type=str2bool, nargs='?',
                        const=True, default=False, required=False,
                help="Uses EarlyStopping"
                "Default uses False")
        subparser_parameters.add_argument("--patience", type=int, default=7, required=False,
                help="How long to wait after last time validation loss improved."
                     "Default: 7")
        subparser_parameters.add_argument("--delta", type=float, default=0, required=False,
                help="Minimum change in the monitored quantity to qualify as an improvement."
                "Default: 0")

        subparsers.add_parser("test_model")

        subparser_parameters.add_argument("--testing_model", type=str2bool, nargs='?',
                        const=True, default=False, required=False,
                help="Test the model performance after training"
                "Default uses False")

        subparser_parameters.add_argument("--test_model_file_location", type=file_path, required=False,
                help="Path to saved model is used for testing (i.e. /path/to/model.pth)."
                 "Default uses the trained model during the training phase")

        subparser_parameters.add_argument("--test_log_dir_location", type=dir_path, required=True,
                help="Location of results of the testing")

        subparser_parameters.add_argument("--detailed_test_result", action='store_true',
                help="Provides deatailed test results, including paths to image files, predicted label, target label, probabilities of classes"
                "Default uses False")

        subparser_parameters.add_argument("--testing_shuffle", action='store_true',
                            help="Shuffle the testing set."
                                 "Default uses False")

        subparser_parameters.add_argument("--test_chunks", nargs="+", type=int,
                default=[2],
                help="Space separated number IDs specifying chunks to use for testing.")


def get_args():
        parser = create_parser()
        args = parser.get_args()
        set_random_seed(args.seed)
        return args

