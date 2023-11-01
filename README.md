# Docker Train

## Usage

```
usage: app.py [-h] --experiment_name EXPERIMENT_NAME --batch_size BATCH_SIZE
              --validation_interval VALIDATION_INTERVAL --epochs EPOCHS
              [--training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]]
              [--validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]]
              [--is_binary] [--subtypes SUBTYPES [SUBTYPES ...]]
              [--patch_pattern PATCH_PATTERN] --chunk_file_location
              CHUNK_FILE_LOCATION --log_dir_location LOG_DIR_LOCATION
              --model_dir_location MODEL_DIR_LOCATION --model_config_location
              MODEL_CONFIG_LOCATION [--patch_location PATCH_LOCATION]
              [--num_patch_workers NUM_PATCH_WORKERS]
              [--num_validation_batches NUM_VALIDATION_BATCHES]
              [--gpu_id GPU_ID] [--seed SEED] [--training_shuffle]
              [--validation_shuffle]

optional arguments:
  -h, --help            show this help message and exit

  --experiment_name EXPERIMENT_NAME
                        experiment name used to name log, model outputs
                         (default: None)

  --batch_size BATCH_SIZE
                        batch size to use on training, validation and test dataset
                         (default: None)

  --validation_interval VALIDATION_INTERVAL
                        the interval of the training loop to start validating model
                         (default: None)

  --epochs EPOCHS       the number of epochs to run model training on training dataset
                         (default: None)

  --training_chunks TRAINING_CHUNKS [TRAINING_CHUNKS ...]
                        space separated number IDs specifying chunks to use for training
                         (default: [0])

  --validation_chunks VALIDATION_CHUNKS [VALIDATION_CHUNKS ...]
                        space separated number IDs specifying chunks to use for validation
                         (default: [1])

  --is_binary           Whether we want to categorize patches by the Tumor/Normal category (true) or by the subtype category (false)
                         (default: False)

  --subtypes SUBTYPES [SUBTYPES ...]
                        space separated words describing subtype=groupping pairs for this study. Example: if doing one-vs-rest on the subtypes MMRD vs P53ABN, P53WT and POLE then the input should be 'MMRD=0 P53ABN=1 P53WT=1 POLE=1'
                         (default: [['MMRD', 0], ['P53ABN', 1], ['P53WT', 2], ['POLE', 3]])

  --patch_pattern PATCH_PATTERN
                        '/' separated words describing the directory structure of the patch paths. The words are 'annotation', 'subtype', 'slide', 'magnification'. A non-multiscale patch can be contained in a directory /path/to/patch/rootdir/Tumor/MMRD/VOA-1234/1_2.png so its patch_pattern is annotation/subtype/slide. A multiscale patch can be contained in a directory /path/to/patch/rootdir/Stroma/P53ABN/VOA-1234/10/3_400.png so its patch pattern is annotation/subtype/slide/magnification
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

  --patch_location PATCH_LOCATION
                        Path to root directory containing dataset patches specified in group or split file (i.e. /path/to/patch/rootdir/). Used by Docker to link the directory.
                         (default: None)

  --num_patch_workers NUM_PATCH_WORKERS
                        Number of loader worker processes to multi-process data loading. Default uses single-process data loading.
                         (default: 0)

  --num_validation_batches NUM_VALIDATION_BATCHES
                        Number of validation patches to use for model validation
                         (default: None)

  --gpu_id GPU_ID       The ID of GPU to select. Default uses GPU with the most free memory
                         (default: None)

  --seed SEED           seed for random shuffle
                         (default: 256)

  --training_shuffle    Shuffle the training set
                         (default: False)

  --validation_shuffle  Shuffle the validation set
                         (default: False)
```
