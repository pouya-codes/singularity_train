import os
import json
import time
import sys
import enum

import yaml
from tqdm import tqdm
from pynvml import *
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import submodule_utils as utils
from submodule_cv import (ChunkLookupException, setup_log_file,
    gpu_selector, PatchHanger)

# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)
nvmlInit()

default_seed = 256
default_num_patch_workers = 0
default_subtypes = {'MMRD':0, 'P53ABN': 1, 'P53WT': 2, 'POLE': 3}
default_patch_pattern = 'annotation/subtype/slide'

class ModelTrainer(PatchHanger):
    """Trains a model
    
    Attributes
    ----------
    experiment_name : str
        Experiment name
    
    train_instance_name : str
        Generated instance name based on experiment name

    batch_size : int
        Batch size to use on training, validation and test dataset

    validation_interval : int
        The interval of the training loop to start validating model
    
    epochs : int
        The number of epochs to run model training on training dataset
    
    training_chunks : list of int
        Space separated number IDs specifying chunks to use for training.

    validation_chunks : list of int
        Space separated number IDs specifying chunks to use for validation.

    is_binary : boolean
        Whether we want to categorize patches by the Tumor/Normal category (true) or by the subtype category (false).

    CategoryEnum : enum.Enum
        The enum representing the categories and is one of (SubtypeEnum, BinaryEnum).

    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths.
        A non-multiscale patch can be contained in a directory /path/to/patch/rootdir/Tumor/MMRD/VOA-1234/1_2.png so its patch_pattern is annotation/subtype/slide.
        A multiscale patch can be contained in a directory /path/to/patch/rootdir/Stroma/P53ABN/VOA-1234/10/3_400.png so its patch pattern is annotation/subtype/slide/magnification

    chunk_file_location : str
        File path of group or split file (aka. chunks) to use (i.e. /path/to/patient_3_groups.json)

    log_dir_location : str
        Path to log directory to save training logs (i.e. /path/to/logs/training/).

    model_dir_location

    model_config_location
    
    model_config : dict
    
    patch_location
    
    num_patch_workers
    
    num_validation_batches

    gpu_id : int
    
    seed : int
    
    training_shuffle : bool
    
    validation_shuffle : bool

    raw_subtypes : dict
    
    raw_patch_pattern : str

    model_file_location : str
        Path to persisted model.
    """

    def __init__(self, config):
        """Initialize training component.

        Arguments
        ---------
        config : argparse.Namespace
            The args passed by user
        """
        self.experiment_name = config.experiment_name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.train_instance_name = f'{self.experiment_name}_{timestamp}'
        # hyperparameters
        self.batch_size = config.batch_size
        self.validation_interval = config.validation_interval
        self.epochs = config.epochs
        # app administration
        self.training_chunks = config.training_chunks
        self.validation_chunks = config.validation_chunks
        self.is_binary = config.is_binary
        self.CategoryEnum = utils.create_category_enum(
                self.is_binary, config.subtypes)
        self.patch_pattern = utils.create_patch_pattern(config.patch_pattern)
        self.chunk_file_location = config.chunk_file_location
        self.log_dir_location = config.log_dir_location
        self.model_dir_location = config.model_dir_location
        self.model_config_location = config.model_config_location
        self.model_config = self.load_model_config()
        # optional
        self.patch_location = config.patch_location # unused
        self.num_patch_workers = config.num_patch_workers
        self.num_validation_batches = config.num_validation_batches
        self.gpu_id = config.gpu_id
        self.seed = config.seed
        self.training_shuffle = config.training_shuffle
        self.validation_shuffle = config.validation_shuffle

        if config.writer_log_dir_location:
            self.writer_log_dir_location = config.writer_log_dir_location
        else:
            self.writer_log_dir_location = os.path.join(
                    self.log_dir_location, self.experiment_name)
        if not os.path.exists(self.writer_log_dir_location):
            os.makedirs(self.writer_log_dir_location)
        self.writer = SummaryWriter(log_dir=self.writer_log_dir_location)
        # raw
        self.raw_subtypes = config.subtypes
        self.raw_patch_pattern = config.patch_pattern
        # model_file_location
        self.model_file_location = os.path.join(self.model_dir_location,
                f'{self.train_instance_name}.pth')

    @classmethod
    def from_config_file(cls, config_file_location):
        pass

    @classmethod
    def from_arguments(cls, config_file_location):
        pass

    def print_parameters(self):
        """Print argument parameters as YAML data to log file.
        The YAML data can be read from the log file by subsequent components.
        """
        payload = yaml.dump({
            ## Arguments
            'experiment_name':     self.experiment_name,
            'batch_size':          self.batch_size,
            'validation_interval': self.validation_interval,
            'epochs':              self.epochs,
            'training_chunks':     self.training_chunks,
            'validation_chunks':   self.validation_chunks,
            'is_binary':           self.is_binary,
            'subtypes':            self.raw_subtypes,
            'patch_pattern': self.raw_patch_pattern,
            'chunk_file_location':   self.chunk_file_location,
            'log_dir_location':    self.log_dir_location,
            'model_dir_location': self.model_dir_location,
            'model_config_location': self.model_config_location,
            'patch_location':    self.patch_location,
            'num_patch_workers': self.num_patch_workers,
            'num_validation_batches': self.num_validation_batches,
            'gpu_id': self.gpu_id,
            'seed':   self.seed,
            'training_shuffle': self.training_shuffle,
            'validation_shuffle': self.validation_shuffle,
            # Generated values
            'instance_name': self.train_instance_name,
            'model_file_location': self.model_file_location,
            'writer_log_dir_location': self.writer_log_dir_location
        })
        print('---') # begin YAML
        print(payload)
        print('...') # end YAML

    def validate(self, model, validation_loader, iter_idx):
        """Runs the validation loop

        Parameters
        ----------
        model : torch.nn.Module
            Model to train

        validation_loader : torch.DataLoader
            Loader for validation set.
        """
        val_idx = 0
        val_loss = 0
        pred_labels = []
        gt_labels = []
        model.model.eval()
        with torch.no_grad():
            for data in validation_loader:
                if self.num_validation_batches is not None \
                        and val_idx >= self.num_validation_batches:
                    break
                cur_data, cur_label, _ = data
                cur_data = cur_data.cuda()
                cur_label = cur_label.cuda()
                logits, pred_prob, output = model.forward(cur_data)
                val_loss += model.get_loss(logits, cur_label, output).item()
                # if self.is_binary:
                #     pred_labels += (pred_prob >=
                #             0.5).type(torch.int).cpu().numpy().tolist()
                # else:
                pred_labels += torch.argmax(pred_prob,
                        dim=1).cpu().numpy().tolist()
                gt_labels += cur_label.cpu().numpy().tolist()
                val_idx += 1
        model.model.train()
        return accuracy_score(gt_labels, pred_labels), (val_loss / val_idx)

    def train(self, model, training_loader, validation_loader):
        """Runs the training loop

        Parameters
        ----------
        model : torch.nn.Module
            Model to train
        
        training_loader : torch.DataLoader
            Loader for training set.

        validation_loader : torch.DataLoader
            Loader for validation set.
        """
        iter_idx = -1
        max_val_acc = float('-inf')
        max_val_acc_idx = -1
        intv_loss = 0
        pred_labels = []
        gt_labels = []
        for epoch in range(self.epochs):
            prefix = f'Training Epoch {epoch}: '
            for data in tqdm(training_loader, desc=prefix, 
                    dynamic_ncols=True, leave=True, position=0):
                iter_idx += 1
                batch_data, batch_labels,_ = data
                batch_data = batch_data.cuda()
                batch_labels = batch_labels.cuda()
                logits, probs, output = model.forward(batch_data)
                model.optimize_parameters(logits, batch_labels, output)
                intv_loss += model.get_current_errors()
                pred_labels += torch.argmax(probs,
                        dim=1).cpu().numpy().tolist()
                gt_labels += batch_labels.cpu().numpy().tolist()
                if iter_idx % self.validation_interval == 0:

                    val_acc, val_loss = self.validate(model, validation_loader, iter_idx)
                    self.writer.add_scalars(f"{self.train_instance_name}/loss",
                            {
                                'validation': val_loss,
                                'test': intv_loss / self.validation_interval
                            }, iter_idx)
                    self.writer.add_scalars(f"{self.train_instance_name}/accuracy",
                            {
                                'validation': val_acc,
                                'test': accuracy_score(gt_labels, pred_labels)
                            }, iter_idx)
                    intv_loss = 0
                    pred_labels = []
                    gt_labels = []
                    if max_val_acc <= val_acc:
                        max_val_acc = val_acc
                        max_val_acc_idx = iter_idx
                        model.save_state(self.model_dir_location,
                                self.train_instance_name,
                                iter_idx, epoch)
                self.writer.flush()
            print(f'\nEpoch: {epoch}')
            print(f'Peak accuracy: {max_val_acc}')
            print(f'Peach accuracy at iteration: {max_val_acc_idx}')
            self.writer.close()

    def build_model(self):
        model = super().build_model()
        print(model.model)
        return model


    def run(self):
        setup_log_file(self.log_dir_location, self.train_instance_name)
        self.print_parameters()
        print(f'Instance name: {self.train_instance_name}')
        gpu_selector(self.gpu_id)
        training_loader = self.create_data_loader(self.training_chunks, color_jitter=True, shuffle=self.training_shuffle)
        validation_loader = self.create_data_loader(self.validation_chunks, shuffle=self.validation_shuffle)
        model = self.build_model()
        self.train(model, training_loader, validation_loader)
