# Singularity Train
```
Development information

Date Created: 22 July 2020
Last Update: 23 July 2020 by Pouya Ahmadvand
Developer: Colin Chen
Version: 1.0
```
### 1. Description ###
This branch is singularity-based version on [docker_train](https://svn.bcgsc.ca/bitbucket/projects/MLOVCA/repos/docker_train/browse) master branch.
### 2. How to Use ###
Follow steps in this [link](https://www.bcgsc.ca/wiki/display/OVCARE/Singularity+on+Numbers).


To build singularity image

```
$singularity build --remote singularity_train.sif Singularityfile.def
```

To run the container afterwards

```
$singularity run --nv singularity_train.sif from-experiment-manifest "path/to/manifest/file/location" 
```

### 3. Parameters ###


```
usage: app.py [-h] [from-experiment-manifest path/to/experiment-manifest-file, from-arguments --experiment_name EXPERIMENT_NAME --batch_size BATCH_SIZE
              --validation_interval VALIDATION_INTERVAL --epochs EPOCHS
              [--training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]]
              [--validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]]
              [--is_binary] [--subtypes SUBTYPES [SUBTYPES ...]]
              [--patch_pattern PATCH_PATTERN] --chunk_file_location
              CHUNK_FILE_LOCATION --log_dir_location LOG_DIR_LOCATION
              --model_dir_location MODEL_DIR_LOCATION --model_config_location
              MODEL_CONFIG_LOCATION 
              [--num_patch_workers NUM_PATCH_WORKERS]
              [--num_validation_batches NUM_VALIDATION_BATCHES]
              [--gpu_id GPU_ID] [--seed SEED] [--training_shuffle]
              [--validation_shuffle]]

Trains a model for patch classification. This process does the training in the following manner:

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
            "amsgrad" : true
        }
    }
}

 (3) For each epoch (specified by --epochs), we train the classifier using all patches in the training set, feeding the classifier a batch of patches (with size specified by --batch_size). At every batch interval (specififed by --validation_interval) we run validation loop and save (or overwrite) the model if it achieves the as of yet highest validation accuracy.

 (4) All argument flags used in `docker_train` are saved to a YAML section in the log file (specified by --log_dir_location). Components using the model like `docker_evaluate` reads this log file to find the saved model and model parameters.

optional arguments:
  -h, --help            show this help message and exit

  --experiment_name EXPERIMENT_NAME
                        Experiment name used to name log, model outputs.
                         (default: None)

  --batch_size BATCH_SIZE
                        Batch size is the number of patches to put in a batch. This flag sets the batch size to use on training, validation and test datasets.
                         (default: None)

  --validation_interval VALIDATION_INTERVAL
                        The interval of the training loop to start validating model.
                         (default: None)

  --epochs EPOCHS       The number of epochs to run model training on training dataset.
                         (default: None)

  --training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]
                        Space separated number IDs specifying chunks to use for training.
                         (default: [0])

  --validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]
                        Space separated number IDs specifying chunks to use for validation.
                         (default: [1])

  --is_binary           Whether we want to categorize patches by the Tumor/Normal category (true) or by the subtype category (false).
                         (default: False)

  --subtypes SUBTYPES [SUBTYPES ...]
                        space separated words describing subtype=groupping pairs for this study. Example: if doing one-vs-rest on the subtypes MMRD vs P53ABN, P53WT and POLE then the input should be 'MMRD=0 P53ABN=1 P53WT=1 POLE=1'
                         (default: {'MMRD': 0, 'P53ABN': 1, 'P53WT': 2, 'POLE': 3})

  --patch_pattern PATCH_PATTERN
                        '/' separated words describing the directory structure of the patch paths. The words are ('annotation', 'subtype', 'slide', 'patch_size', 'magnification'). A non-multiscale patch can be contained in a directory /path/to/patch/rootdir/Tumor/MMRD/VOA-1234/1_2.png so its patch_pattern is annotation/subtype/slide. A multiscale patch can be contained in a directory /path/to/patch/rootdir/Stroma/P53ABN/VOA-1234/10/3_400.png so its patch pattern is annotation/subtype/slide/magnification
                         (default: annotation/subtype/slide)

  --chunk_file_location CHUNK_FILE_LOCATION
                        File path of group or split file (aka. chunks) to use (i.e. /path/to/patient_3_groups.json)
                         (default: None)

  --log_dir_location LOG_DIR_LOCATION
                        Path to log directory to save training logs (i.e. /path/to/logs/training/).
                         (default: None)

  --model_dir_location MODEL_DIR_LOCATION
                        Path to model directory to save trained model (i.e. /path/to/model/).
                         (default: None)

  --model_config_location MODEL_CONFIG_LOCATION
                        Path to model config JSON (i.e. /path/to/model_config.json).
                         (default: None)

  --num_patch_workers NUM_PATCH_WORKERS
                        Number of loader worker processes to multi-process data loading. Default uses single-process data loading.
                         (default: 0)

  --num_validation_batches NUM_VALIDATION_BATCHES
                        Number of validation batches to use for model validation. Default uses all validation batches for each validation loop.
                         (default: None)

  --gpu_id GPU_ID       The ID of GPU to select. Default uses GPU with the most free memory.
                         (default: None)
                         
  --number_of_gpus      The number of GPUs to use. Default uses a GPU with the most free memory.
                         (default: 1)

  --seed SEED           Seed for random shuffle.
                         (default: 256)

  --training_shuffle    Shuffle the training set.
                         (default: False)

  --validation_shuffle  Shuffle the validation set.
                         (default: False)

  --use_early_stopping  Use EarlyStopping.
                         (Default uses False)

  --patience            How long to wait after last time validation loss improved.
                         (Default: 7)

  --delta               Minimum change in the monitored quantity to qualify as an improvement.
                         (Default: 0)

  --testing_model       Test the model performance after training.
                         (Default: False)

  --test_model_file_location TEST_MODEL_FILE_LOCATION
                        Path to saved model is used for testing (i.e. /path/to/model.pth). Default uses the trained model during the training phase
                         (default: None)

  --test_log_dir_location
                        Location of results of the testing

  --detailed_test_result
                        Provides deatailed test results, including paths to image files, predicted label, target label, probabilities of classes
                         (Default: False)

  --testing_shuffle     Shuffle the testing set.
                         (Default: False)

  --test_chunks         Space separated number IDs specifying chunks to use for testing.
                         (Default: [2])

TODO: See JIRA tickets for updating docker_{train, evaluate}
```
