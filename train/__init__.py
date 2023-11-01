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

import submodule_utils as utils
from submodule_cv import (ChunkLookupException, setup_log_file,
    gpu_selector, PatchHanger)

# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)
nvmlInit()

class ModelTrainer(PatchHanger):
    '''Trains a model
    
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
    
    TODO: finish attributes
    training_chunks
    self.validation_chunks = validation_chunks
    self.is_binary = is_binary
    self.CategoryEnum = utils.create_category_enum(
            self.is_binary, subtypes)
    self.patch_pattern = utils.create_patch_pattern(patch_pattern)
    self.chunk_file_location = chunk_file_location
    self.log_dir_location = log_dir_location
    self.model_dir_location = model_dir_location
    self.model_config_location = model_config_location
    self.model_config = self.load_model_config()
    # optional
    self.patch_location = patch_location # unused
    self.num_patch_workers = num_patch_workers
    self.num_validation_batches = num_validation_batches
    self.gpu_id = gpu_id
    self.seed = seed
    self.training_shuffle = training_shuffle
    self.validation_shuffle = validation_shuffle
    # raw
    self.raw_subtypes = subtypes
    self.raw_patch_pattern = patch_pattern
    # model_file_location
    self.model_file_location = os.path.join(self.model_dir_location,
            f'{self.train_instance_name}.pth')
    '''

    def __init__(self,
            experiment_name,
            batch_size,
            validation_interval,
            epochs,
            training_chunks,
            validation_chunks,
            is_binary,
            subtypes,
            patch_pattern,
            chunk_file_location,
            log_dir_location,
            model_dir_location,
            model_config_location,
            patch_location=None,
            num_patch_workers=0,
            num_validation_batches=None,
            gpu_id=None,
            seed=256,
            training_shuffle=False,
            validation_shuffle=False):
        '''
        Changes:
        test_name => experiment_name
        validation_frequency => validation_interval
        epochs
        log_folder_location => log_dir_location
        model_save_location => model_dir_location
        patch_workers => num_patch_workers
        path_value_to_index => subtypes
        model_config => model_config_location
        '''
        self.experiment_name = experiment_name
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.train_instance_name = f'{self.experiment_name}_{timestamp}'
        # hyperparameters
        self.batch_size = batch_size
        self.validation_interval = validation_interval
        self.epochs = epochs
        # app administration
        self.training_chunks = training_chunks
        self.validation_chunks = validation_chunks
        self.is_binary = is_binary
        self.CategoryEnum = utils.create_category_enum(
                self.is_binary, subtypes)
        self.patch_pattern = utils.create_patch_pattern(patch_pattern)
        self.chunk_file_location = chunk_file_location
        self.log_dir_location = log_dir_location
        self.model_dir_location = model_dir_location
        self.model_config_location = model_config_location
        self.model_config = self.load_model_config()
        # optional
        self.patch_location = patch_location # unused
        self.num_patch_workers = num_patch_workers
        self.num_validation_batches = num_validation_batches
        self.gpu_id = gpu_id
        self.seed = seed
        self.training_shuffle = training_shuffle
        self.validation_shuffle = validation_shuffle
        # raw
        self.raw_subtypes = subtypes
        self.raw_patch_pattern = patch_pattern
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
        '''Print argument parameters
        '''
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
            'model_file_location': self.model_file_location
        })
        print('---') # begin YAML
        print(payload)
        print('...') # end YAML

    def validate(self, model, validation_loader):
        pred_labels = []
        gt_labels = []
        model.model.eval()
        with torch.no_grad():
            for val_idx, data in enumerate(validation_loader):
                if self.num_validation_batches is not None \
                        and val_idx >= self.num_validation_batches:
                    break
                cur_data, cur_label = data
                cur_data = cur_data.cuda()
                cur_label = cur_label.cuda()
                _, pred_prob, _ = model.forward(cur_data)
                if self.is_binary:
                    pred_labels += (pred_prob >=
                            0.5).type(torch.int).cpu().numpy().tolist()
                else:
                    pred_labels += torch.argmax(pred_prob,
                            dim=1).cpu().numpy().tolist()
                gt_labels += cur_label.cpu().numpy().tolist()
                
        model.model.train()
        return accuracy_score(gt_labels, pred_labels)

    def train(self, model, training_loader, validation_loader):
        iter_idx = -1
        max_val_acc = float('-inf')
        max_val_acc_idx = -1
        for epoch in range(self.epochs):
            prefix = f'Training Epoch {epoch}: '
            for data in tqdm(training_loader, desc=prefix, 
                    dynamic_ncols=True, leave=True, position=0):
                iter_idx += 1
                batch_data, batch_labels = data
                batch_data = batch_data.cuda()
                batch_labels = batch_labels.cuda()
                logits, probs, output = model.forward(batch_data)
                model.optimize_parameters(logits, batch_labels, output)
                if iter_idx % self.validation_interval == 0:
                    val_acc = self.validate(model, validation_loader)
                    if max_val_acc <= val_acc:
                        max_val_acc = val_acc
                        max_val_acc_idx = iter_idx
                        model.save_state(self.model_dir_location,
                                self.train_instance_name,
                                iter_idx, epoch)
            print(f'\nEpoch: {epoch}')
            print(f'Peak accuracy: {max_val_acc}')
            print(f'Peach accuracy at iteration: {max_val_acc_idx}')

    def run(self):
        setup_log_file(self.log_dir_location, self.train_instance_name)
        self.print_parameters()
        print(f'Instance name: {self.train_instance_name}')
        gpu_selector(self.gpu_id)
        training_loader = self.create_data_loader(self.training_chunks,
                color_jitter=True, shuffle=self.training_shuffle)
        validation_loader = self.create_data_loader(self.validation_chunks,
                shuffle=self.validation_shuffle)
        model = self.build_model()
        self.train(model, training_loader, validation_loader)
