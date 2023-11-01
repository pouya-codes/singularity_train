# Singularity Train - a python module to train deep learning models on histopathology data

### Development Information ###

```
**Before running any experiment to be sure you are using the latest commits of all modules run the following script:**
```
(cd /projects/ovcare/classification/singularity_modules ; ./update_modules.sh --bcgsc-pass your/bcgsc/path)
```

### Usage ###
```

usage: app.py [-h] {from-experiment-manifest,from-arguments} ...

Trains a model for patch classification. This process does the training in the following manner:

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

positional arguments:
  {from-experiment-manifest,from-arguments}
                        Choose whether to use arguments from experiment manifest or from commandline
    from-experiment-manifest
                        Use experiment manifest

    from-arguments      Use arguments

optional arguments:
  -h, --help            show this help message and exit

usage: app.py from-experiment-manifest [-h] [--component_id COMPONENT_ID]
                                       experiment_manifest_location

positional arguments:
  experiment_manifest_location

optional arguments:
  -h, --help            show this help message and exit

  --component_id COMPONENT_ID

usage: app.py from-arguments [-h] --experiment_name EXPERIMENT_NAME
                             [--train_model [TRAIN_MODEL]] --batch_size
                             BATCH_SIZE --validation_interval
                             VALIDATION_INTERVAL --epochs EPOCHS
                             [--training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]]
                             [--validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]]
                             [--is_binary]
                             [--subtypes SUBTYPES [SUBTYPES ...]]
                             [--patch_pattern PATCH_PATTERN]
                             --chunk_file_location CHUNK_FILE_LOCATION
                             --log_dir_location LOG_DIR_LOCATION
                             --model_dir_location MODEL_DIR_LOCATION
                             [--save_model_for_export] --model_config_location
                             MODEL_CONFIG_LOCATION
                             [--num_patch_workers NUM_PATCH_WORKERS]
                             [--num_validation_batches NUM_VALIDATION_BATCHES]
                             [--gpu_id GPU_ID]
                             [--number_of_gpus NUMBER_OF_GPUS] [--seed SEED]
                             [--training_shuffle] [--validation_shuffle]
                             [--progressive_resizing PROGRESSIVE_RESIZING [PROGRESSIVE_RESIZING ...]]
                             [--scheduler_step {epoch,batch}]
                             [--writer_log_dir_location WRITER_LOG_DIR_LOCATION]
                             {freeze_training,early_stopping,representation_learning,test_model}
                             ...

positional arguments:
  {freeze_training,early_stopping,representation_learning,test_model}

optional arguments:
  -h, --help            show this help message and exit

  --experiment_name EXPERIMENT_NAME
                        Experiment name used to name log, model outputs.
                         (default: None)

  --train_model [TRAIN_MODEL]
                        Train the model or just test the model
                         (default: True)

  --batch_size BATCH_SIZE
                        Batch size is the number of patches to put in a batch. This flag sets the batch size to use on training, validation and test datasets.
                         (default: None)

  --validation_interval VALIDATION_INTERVAL
                        The interval of the training loop to start validating model. For validation only once in each epoch, set this value to -1.
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

  --save_model_for_export
                        Whether we want to save the entire model for export
                         (default: False)

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

  --number_of_gpus NUMBER_OF_GPUS
                        The number of GPUs to use. Default uses a GPU with the most free memory.
                         (default: 1)

  --seed SEED           Seed for random shuffle.
                         (default: 256)

  --training_shuffle    Shuffle the training set.
                         (default: False)

  --validation_shuffle  Shuffle the validation set.
                         (default: False)

  --progressive_resizing PROGRESSIVE_RESIZING [PROGRESSIVE_RESIZING ...]
                        One of the techniques that enables the model to be trained on different input sizes. The inputs are resized to the selected value and trained on them in order of the size. For example, [32, 256, 512] means that the model will be trained first on 32, then 256, and at last on 512.
                         (default: [-1])

  --scheduler_step {epoch,batch}
                        If a scheduler is defined in the config file, it enables the step for that. It should be 'epoch' or 'batch'. It depends on the scheduler, for example OneCycleLR needs to be stepped in batch.
                         (default: None)

  --writer_log_dir_location WRITER_LOG_DIR_LOCATION
                        Directory in log_dir_location to put TensorBoard logs.Default uses log_dir_location/experiment_name.
                         (default: None)

usage: app.py from-arguments freeze_training [-h]
                                             [--use_freeze_training [USE_FREEZE_TRAINING]]
                                             [--freeze_epochs FREEZE_EPOCHS]
                                             [--unfreeze_epochs UNFREEZE_EPOCHS]
                                             [--base_lr BASE_LR]
                                             [--lr_mult LR_MULT]
                                             [--use_scheduler]
                                             [--use_early_stopping [USE_EARLY_STOPPING]]
                                             [--patience PATIENCE]
                                             [--delta DELTA]
                                             [--use_representation_learning [USE_REPRESENTATION_LEARNING]]
                                             [--model_path_location MODEL_PATH_LOCATION]
                                             [--train_feature_extractor [TRAIN_FEATURE_EXTRACTOR]]
                                             [--testing_model [TESTING_MODEL]]
                                             [--test_model_file_location TEST_MODEL_FILE_LOCATION]
                                             [--old_version]
                                             --test_log_dir_location
                                             TEST_LOG_DIR_LOCATION
                                             [--detailed_test_result]
                                             [--testing_shuffle]
                                             [--test_chunks TEST_CHUNKS [TEST_CHUNKS ...]]
                                             [--calculate_slide_level_accuracy [CALCULATE_SLIDE_LEVEL_ACCURACY]]
                                             [--slide_level_accuracy_threshold SLIDE_LEVEL_ACCURACY_THRESHOLD]
                                             [--slide_level_accuracy_verbose [SLIDE_LEVEL_ACCURACY_VERBOSE]]
                                             [--generate_heatmaps]
                                             [--heatmaps_dir_location HEATMAPS_DIR_LOCATION]
                                             [--slides_location SLIDES_LOCATION]

optional arguments:
  -h, --help            show this help message and exit

Freeze Training:
  Enable parameters for training with both freeze or unfreeze setup. The models are divided into two parts of Feature Extraction and Classifier. In freeze, batch normalization layers of Feature Extraction and all the layers of Classifier  are trained. In unfreeze, all layers are trained. Freeze Training means that train the second part (classsifier) for some epochs, and then train the whole model toghether.

  --use_freeze_training [USE_FREEZE_TRAINING]
                        Enable Freeze Training
                         (default: False)

  --freeze_epochs FREEZE_EPOCHS
                        Number of epochs model should be tranied in freeze mode.
                         (default: 1)

  --unfreeze_epochs UNFREEZE_EPOCHS
                        Number of epochs model should be tranied in unfreeze mode.
                         (default: 5)

  --base_lr BASE_LR     Base learning rate. In freeze part, the classifier learning part will be base_lr, and the feature_extraction will be base_lr/10.
                         (default: 0.002)

  --lr_mult LR_MULT     In unfreeze part, the classifier learning part will be base_lr/2, and the feature_extraction will be base_lr/(2*lr_mult).
                         (default: 100)

  --use_scheduler       Using scheduler; If a scheduler is defined in the config file, it will use that. Otherwise, the model will use OneCycleLR.
                         (default: False)

Early Stopping:
  Enable early stopping.

  --use_early_stopping [USE_EARLY_STOPPING]
                        Enable Early Stopping
                         (default: False)

  --patience PATIENCE   How long to wait after last time validation loss improved.
                         (default: 7)

  --delta DELTA         Minimum change in the monitored quantity to qualify as an improvement.
                         (default: 0)

Representation Learning:
  Use trained model on representation learning for classification.

  --use_representation_learning [USE_REPRESENTATION_LEARNING]
                        Enable representation learning
                         (default: False)

  --model_path_location MODEL_PATH_LOCATION
                        Path to saved model is used for representation (i.e. /path/to/model.pth).
                         (default: None)

  --train_feature_extractor [TRAIN_FEATURE_EXTRACTOR]
                        By defuault, only classifier part should be trained. By setting this flag, the feature extractor will be trained too.
                         (default: False)

Testing:
  Enable parameters for testing.

  --testing_model [TESTING_MODEL]
                        Test the model performance after training
                         (default: False)

  --test_model_file_location TEST_MODEL_FILE_LOCATION
                        Path to saved model is used for testing (i.e. /path/to/model.pth).Default uses the trained model during the training phase
                         (default: None)

  --old_version         Convert trained model on previous version to the current one
                         (default: False)

  --test_log_dir_location TEST_LOG_DIR_LOCATION
                        Location of results of the testing
                         (default: None)

  --detailed_test_result
                        Provides deatailed test results, including paths to image files, predicted label, target label, probabilities of classesDefault uses False
                         (default: False)

  --testing_shuffle     Shuffle the testing set.
                         (default: False)

  --test_chunks TEST_CHUNKS [TEST_CHUNKS ...]
                        Space separated number IDs specifying chunks to use for testing.
                         (default: [2])

  --calculate_slide_level_accuracy [CALCULATE_SLIDE_LEVEL_ACCURACY]
                        Wether calculate slide level accuracy
                         (default: False)

  --slide_level_accuracy_threshold SLIDE_LEVEL_ACCURACY_THRESHOLD
                        Minimum threshold that the patch labels probabilities should pass to be considered in the slide level voting. Default: 1/number_of_classes
                         (default: 0)

  --slide_level_accuracy_verbose [SLIDE_LEVEL_ACCURACY_VERBOSE]
                        Verbose the detail for slide level accuracy
                         (default: False)

  --generate_heatmaps   Generate the overlay heatmaps that can be used to visualize the results on cPathPortal
                         (default: False)

  --heatmaps_dir_location HEATMAPS_DIR_LOCATION
                        Location of generated heatmaps files
                         (default: None)

  --slides_location SLIDES_LOCATION
                        Path to the slides used to extract and generate data splits
                         (default: None)
                         
  --gradcam				Generate the grad cam image files
                        (default: False)
                        
  --gradcam_dir_location GRADCAM_DIR_LOCATION
  						Location of generated grad cam image
                        (default: None)
                        
  --slides_location SLIDES_LOCATION
  						Path to the slides used to create gradCAM output
                        (default: None)
                        
  --gradcam_h5			Generate the overlay grad cam h5 files that can be used to visualize the results on cPathPortal
                        (default: False)

  

```


Note: `freeze_training` subparser MUST be used in your manifest. Due to more readability, the parser is
defined in this way (having multiple subparserss instead of just one).


### Config File ###
Model definition and augmentations are defined in the `config.json` file. It is divided into:
1. `model`:
1.1. `num_subtypes`: number of output neurons (number of classes).
1.2. `base_model`: the baseline model.
1.3. `pretrained`: {true,false} -> if true, uses ImageNet trained weights.
1.4. `last_layers`: {short,long} -> short means only changing last layer neurons to be compatible to our dataset, but long means adding more layers to fully connected section.
1.5. `concat_pool`: {true,false} -> if true, uses concatination of max and average pooling. If not, use only max pooling. (Note: it only affects when the last_layer is long!)
2. `normalize`:
2.1. `use_normalize`: {true,false} -> if true, normalize the dataset based on provided mean and std. Otherwise, use 0.5 as mean and std for each channel.
2.2. `mean`: average for each channel
2.3. `std`: standard deviation for each channel
3. `augmentation`:
3.1. `use_augmentation`: {true,false} -> whether use augmentation or not at all
3.2. `flip`: {true,false} -> if true, add vertical and horizantal flip to the augmentation list.
3.3. `color_jitter`: {true,false} -> if true, add color jitter to the augmentation list.
3.3. `rotation`: {true,false} -> if true, add 20 degree rotation to the augmentation list.
3.3. `crop`: {int} -> if set, add crop of determied size to the augmentation list.
3.3. `resize`: {int} -> if set, add resize of determied size to the augmentation list.
3.4. `size_jitter`:
3.4.1. `use_size_jitter`: {true,false} -> if true, add size_jitter to the augmentation list.
3.4.2. `ratio`: {float} -> ratio of the original image size
3.4.3. `probability`: {float} -> probability of doing this augmentation
<<<<<<< HEAD
3.4.4. `color`: {white,black} -> the color of padding when use ratio less than 1.
=======
3.4.4. `color`: {white,black} -> the color of padding when the image is resized smaller
>>>>>>> refs/rewritten/Added-Gradcam
3.4.5. `dynamic_bool`: {true, false} -> if true, choose a random value between size*(1-ratio) and size*(1+ratio) inclusive to resize the image. If false, only choose either size*(1-ratio) or size*(1+ratio) for resizing. Default false.
3.5. `cut_out`:
3.5.1. `use_num_cut`: {true,false} -> if true, add num_cut to the augmentation list.
3.5.2. `num_cut`: {int} -> number of cutouts
3.5.3. `size_cut`: {int} -> size of each cutout in pixels
3.5.4. `color_cut`: {white,black} -> the color of cutouts
4. `use_weighted_loss`:
4.1. `use_weighted_loss`: {true,false} -> if true, use weighted loss.
4.2. `weight`: {array of floats} -> weights used for each class in the loss
5. `use_weighted_sampler`: {true,false} -> if true, use a weighted sampler that sample the data with the weights 1/num_class (for imbalancing).
6. `use_balanced_sampler`: {true,false} -> if true, use a balanced sampler that in each batch, we will have same number of data from each class (for imbalancing).
7. `mix_up`:
7.1. `use_mix_up`: {true,false} -> if true, use mix_up technique.
7.2. `alpha`: {float} -> betha distribution (recommended to set as 0.4)
8. `freeze`: useless (TODO: remove this flag)
9. `continue_train`: useless (TODO: remove this flag)
10. `optimizer`: Optimizer settings ..
11. `scheduler`: Scheduler settings ..

Note: If both resize and crop are added, first resize is applied and then crop.

