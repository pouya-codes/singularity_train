import argparse
import os
from train import ModelTrainer

description=None
epilog=None

def dir_path(s):
    if os.path.isdir(s):
        return s
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{s} is not a valid path")

def file_path(s):
    if os.path.isfile(s):
        return s
    else:
        raise argparse.ArgumentTypeError(f"readable_file:{s} is not a valid path")

def dict_vals(kv):
    try:
        k, v = kv.split("=")
    except:
        raise argparse.ArgumentTypeError(f"value {kv} is not separated by one '='")
    try:
        v = int(v)
    except:
        raise argparse.ArgumentTypeError(f"right side of {kv} should be int")
    return (k, v)

def make_subtype_dict(ll):
    return {k: v for (k, v) in ll}

class CustomHelpFormatter(
        argparse.RawTextHelpFormatter, 
        argparse.ArgumentDefaultsHelpFormatter):
    def add_argument(self, action):
        action.help += '\n'
        super().add_argument(action)

    def _format_action(self, action):
        s = super()._format_action(action)
        return s + '\n'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=description, epilog=epilog,
            formatter_class=CustomHelpFormatter)
    
    parser.add_argument("--experiment_name", type=str, required=True,
            help="experiment name used to name log, model outputs")

    parser.add_argument("--batch_size", type=int, required=True,
            help="batch size to use on training, validation and test dataset")
    
    parser.add_argument("--validation_interval", type=int, required=True,
            help="the interval of the training loop to start validating model")
    
    parser.add_argument("--epochs", type=int, required=True,
            help="the number of epochs to run model training on training dataset")

    parser.add_argument("--training_chunks", nargs="+", type=int,
            default=[0],
            help="space separated number IDs specifying chunks to use for training")
    
    parser.add_argument("--validation_chunks", nargs="+", type=int,
            default=[1],
            help="space separated number IDs specifying chunks to use for validation")

    parser.add_argument("--is_binary", action='store_true',
            help='Whether we want to categorize patches by the Tumor/Normal category (true) '
            'or by the subtype category (false)')

    parser.add_argument("--subtypes", nargs='+', type=dict_vals,
            default=[['MMRD', 0], ['P53ABN', 1], ['P53WT', 2], ['POLE', 3]],
            help="space separated words describing subtype=groupping pairs for this study. "
            "Example: if doing one-vs-rest on the subtypes MMRD vs P53ABN, P53WT and POLE then "
            "the input should be 'MMRD=0 P53ABN=1 P53WT=1 POLE=1'")
    
    parser.add_argument("--patch_pattern", type=str,
            default='annotation/subtype/slide',
            help="'/' separated words describing the directory structure of the "
            "patch paths. The words are 'annotation', 'subtype', 'slide', "
            "'magnification'. A non-multiscale patch can be contained in a directory "
            "/path/to/patch/rootdir/Tumor/MMRD/VOA-1234/1_2.png so its patch_pattern is "
            "annotation/subtype/slide. A multiscale patch can be contained in a "
            "directory /path/to/patch/rootdir/Stroma/P53ABN/VOA-1234/10/3_400.png so "
            "its patch pattern is annotation/subtype/slide/magnification")
    
    parser.add_argument("--chunk_file_location", type=file_path, required=True,
            help="File path of group or split file (aka. chunks) to use "
            "(i.e. /path/to/patient_3_groups.json)")
    
    parser.add_argument("--patch_location", type=dir_path, required=True,
            help="Path to root directory containing dataset patches specified in "
            "group or split file (i.e. /path/to/patch/rootdir/). Used by Docker "
            "to link the directory.")

    parser.add_argument("--log_dir_location", type=dir_path, required=True,
            help="Path to log directory to save training logs (i.e. "
            "/path/to/logs/training/).")
    
    parser.add_argument("--model_dir_location", type=dir_path, required=True,
            help="Path to model directory to save trained model (i.e. /path/to/model/).")

    parser.add_argument("--model_config_location", type=str, required=True,
            help="Path to model config JSON (i.e. /path/to/model_config.json).")

    parser.add_argument("--num_patch_workers", type=int, default=0,
            help="Number of loader worker processes to multi-process data loading. "
            "Default uses single-process data loading.")
    
    parser.add_argument("--num_validation_batches", type=int, required=False,
            help="Number of validation patches to use for model validation")
    
    parser.add_argument("--gpu_id", type=int, required=False,
            help="The ID of GPU to select. Default uses GPU with the most free memory")

    parser.add_argument("--seed", type=int, default=256,
            help='seed for random shuffle')

    parser.add_argument("--training_shuffle", action='store_true',
            help="Shuffle the training set")

    parser.add_argument("--validation_shuffle", action='store_true',
            help="Shuffle the validation set")

    args = parser.parse_args()
    subtypes = make_subtype_dict(args.subtypes)
    mt = ModelTrainer(
            args.experiment_name,
            args.batch_size,
            args.validation_interval,
            args.epochs,
            args.training_chunks,
            args.validation_chunks,
            args.is_binary,
            subtypes,
            args.patch_pattern,
            args.chunk_file_location,
            args.patch_location,
            args.log_dir_location,
            args.model_dir_location,
            args.model_config_location,
            num_patch_workers=args.num_patch_workers,
            num_validation_batches=args.num_validation_batches,
            gpu_id=args.gpu_id,
            seed=args.seed,
            training_shuffle=args.training_shuffle,
            validation_shuffle=args.validation_shuffle)
    mt.run()
