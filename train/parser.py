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
    "model" :{
        "num_subtypes" : 2,
        "base_model" : "resnet18",
        "pretrained" : false,
        "last_layers": "short",
        "concat_pool": true
    },
    "normalize" : {
        "use_normalize" : false,
        "mean" : [ 0.7371, 0.6904, 0.8211 ],
        "std" : [ 0.0974, 0.0945, 0.0523 ]
    },
    "augmentation" : {
        "use_augmentation": false,
        "flip": true,
        "color_jitter": true,
        "rotation": true,
        "cut_out": {
            "num_cut": 2,
            "size_cut": 100,
            "color_cut": "white"
        }
    },
    "use_weighted_loss" : {
        "use_weighted_loss" : false,
        "weight": [2, 1]
    },
    "use_weighted_sampler" : false,
    "use_balanced_sampler" : false,
    "mix_up" : {
        "use_mix_up" : false,
        "alpha": 0.4
    },
    "freeze": -1,
    "continue_train": false,
    "optimizer" : {
        "type" : "Adam",
        "parameters" : {
            "lr" : 0.00001,
            "amsgrad" : true,
            "weight_decay" : 0.0005
        }
    },
    "scheduler" : {
        "type" : "OneCycleLR",
        "parameters" : {
            "max_lr" : 0.01,
            "steps_per_epoch" : 1202,
            "epochs" : 10,
            "three_phase": true
        }
    }
}

    (2.1) If you do not want to use scheduler, remove it from the JSON file.

 (3) For each epoch (specified by --epochs), we train the classifier using all patches in the training set, feeding the classifier a batch of patches (with size specified by --batch_size). At every batch interval (specififed by --validation_interval) we run validation loop and save (or overwrite) the model if it achieves the as of yet highest validation accuracy.
"""

epilog="""
"""

@manifest_arguments(default_component_id="try_me", description=description, epilog=epilog)
def create_parser(parser):

        parser.add_argument("--experiment_name", type=str, required=True,
                help="Experiment name used to name log, model outputs.")

        parser.add_argument("--train_model", type=str2bool, nargs='?',
                const=True, default=True,
                help="Train the model or just test the model")

        parser.add_argument("--batch_size", type=int, required=True,
                help="Batch size is the number of patches to put in a batch. "
                "This flag sets the batch size to use on training, validation and test datasets.")

        parser.add_argument("--validation_interval", type=int, required=True,
                help="The interval of the training loop to start validating model."
                " For validation only once in each epoch, set this value to -1.")

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

        parser.add_argument("--num_validation_batches", type=int,
                help="Number of validation batches to use for model validation. "
                "Default uses all validation batches for each validation loop.")

        parser.add_argument("--gpu_id", type=int,
                help="The ID of GPU to select. Default uses GPU with the most free memory.")

        parser.add_argument("--number_of_gpus", type=int, default=1,
                help="The number of GPUs to use. Default uses a GPU with the most free memory.")

        parser.add_argument("--seed", type=int,
                default=DEAFULT_SEED,
                help="Seed for random shuffle.")

        parser.add_argument("--training_shuffle", action='store_true',
                help="Shuffle the training set.")

        parser.add_argument("--validation_shuffle", action='store_true',
                help="Shuffle the validation set.")

        parser.add_argument("--progressive_resizing", nargs="+", type=int,
                default=[-1],
                help="One of the techniques that enables the model to be trained on different input sizes. "
                "The inputs are resized to the selected value and trained on them in order of the size. "
                "For example, [32, 256, 512] means that the model will be trained first on 32, then 256, and at last on 512.")

        parser.add_argument("--scheduler_step", type=str, required=False,
                choices=['epoch', 'batch'],
                help="If a scheduler is defined in the config file, it enables the step for that. "
                "It should be 'epoch' or 'batch'. It depends on the scheduler, "
                "for example OneCycleLR needs to be stepped in batch.")

        parser.add_argument("--writer_log_dir_location", type=dir_path,
                help="Directory in log_dir_location to put TensorBoard logs."
                "Default uses log_dir_location/experiment_name.")

        subparser = parser.add_subparsers(dest="subparser", required=True)
        subparser_parameters = subparser.add_parser("freeze_training")

        freeze_train_parameters = subparser_parameters.add_argument_group('Freeze Training',
                "Enable parameters for training with both freeze or unfreeze setup. "
                "The models are divided into two parts of Feature Extraction and Classifier. "
                "In freeze, batch normalization layers of Feature Extraction and all the layers of Classifier "
                " are trained. In unfreeze, all layers are trained. "
                "Freeze Training means that train the second part (classsifier) for some epochs, and then train the whole model toghether.")

        freeze_train_parameters.add_argument("--use_freeze_training", type=str2bool, nargs='?',
                const=True, default=False,
                help="Enable Freeze Training")

        freeze_train_parameters.add_argument("--freeze_epochs", type=int, default=1,
                help="Number of epochs model should be tranied in freeze mode.")

        freeze_train_parameters.add_argument("--unfreeze_epochs", type=int, default=5,
                help="Number of epochs model should be tranied in unfreeze mode.")

        freeze_train_parameters.add_argument("--base_lr", type=float, default=2e-3,
                help="Base learning rate. In freeze part, the classifier learning part will be base_lr, and the feature_extraction will be base_lr/10.")

        freeze_train_parameters.add_argument("--lr_mult", type=int, default=100,
                help="In unfreeze part, the classifier learning part will be base_lr/2, and the feature_extraction will be base_lr/(2*lr_mult).")

        freeze_train_parameters.add_argument("--use_scheduler", action='store_true',
                help="Using scheduler; If a scheduler is defined in the config file, it will use that. Otherwise, the model will use OneCycleLR.")

        subparser.add_parser("early_stopping")
        early_stop_parameters = subparser_parameters.add_argument_group('Early Stopping',
                'Enable early stopping.')
        early_stop_parameters.add_argument("--use_early_stopping", type=str2bool, nargs='?',
                const=True, default=False,
                help="Enable Early Stopping")

        early_stop_parameters.add_argument("--patience", type=int, default=7,
                help="How long to wait after last time validation loss improved.")

        early_stop_parameters.add_argument("--delta", type=float, default=0,
                help="Minimum change in the monitored quantity to qualify as an improvement.")

        subparser.add_parser("representation_learning")
        repr_learning_parameters = subparser_parameters.add_argument_group('Representation Learning',
                'Use trained model on representation learning for classification.')
        repr_learning_parameters.add_argument("--use_representation_learning", type=str2bool, nargs='?',
                const=True, default=False,
                help="Enable representation learning")

        repr_learning_parameters.add_argument("--model_path_location", type=file_path,
                help="Path to saved model is used for representation (i.e. /path/to/model.pth).")

        repr_learning_parameters.add_argument("--train_feature_extractor", type=str2bool, nargs='?',
                const=True, default=False,
                help="By defuault, only classifier part should be trained. By setting "
                "this flag, the feature extractor will be trained too.")

        subparser.add_parser("test_model")
        test_parameters = subparser_parameters.add_argument_group('Testing',
                'Enable parameters for testing.')

        test_parameters.add_argument("--testing_model", type=str2bool, nargs='?',
                const=True, default=False,
                help="Test the model performance after training")

        test_parameters.add_argument("--test_model_file_location", type=file_path,
                help="Path to saved model is used for testing (i.e. /path/to/model.pth)."
                 "Default uses the trained model during the training phase")

        test_parameters.add_argument("--old_version", action='store_true',
                help="Convert trained model on previous version to the current one")

        test_parameters.add_argument("--test_log_dir_location", type=dir_path, required=True,
                help="Location of results of the testing")

        test_parameters.add_argument("--detailed_test_result", action='store_true',
                help="Provides deatailed test results, including paths to image files, predicted label, target label, probabilities of classes"
                "Default uses False")

        test_parameters.add_argument("--testing_shuffle", action='store_true',
                help="Shuffle the testing set.")

        test_parameters.add_argument("--test_chunks", nargs="+", type=int,
                default=[2],
                help="Space separated number IDs specifying chunks to use for testing.")

        test_parameters.add_argument("--calculate_slide_level_accuracy", type=str2bool, nargs='?',
                const=True, default=False,
                help="Wether calculate slide level accuracy")

        test_parameters.add_argument("--slide_level_accuracy_threshold", type=float, default=0,
                help="Minimum threshold that the patch labels probabilities should pass to be considered in the slide level voting."
                " Default: 1/number_of_classes")

        test_parameters.add_argument("--slide_level_accuracy_verbose", type=str2bool, nargs='?',
                const=True, default=False,
                help="Verbose the detail for slide level accuracy")

        test_parameters.add_argument("--generate_heatmaps", action='store_true',
                help="Generate the overlay heatmaps that can be used to visualize the results on cPathPortal")

        test_parameters.add_argument("--heatmaps_dir_location", type=dir_path, required=False,
                help="Location of generated heatmaps files")

        test_parameters.add_argument("--gradcam", action='store_true', default= False,
                help="Generate the grad cam image files")

        test_parameters.add_argument("--gradcam_dir_location", type=dir_path, required=False,
                help="Location of generated grad cam image")

        test_parameters.add_argument("--slides_location", type=dir_path, required=False,
                help="Path to the slides used to extract and generate data splits")

        test_parameters.add_argument("--gradcam_h5", action='store_true', default= False,
                help="Generate the overlay grad cam h5 files that can be used to visualize the results on cPathPortal")

def get_args():
        parser = create_parser()
        args = parser.get_args()
        set_random_seed(args.seed)
        return args
