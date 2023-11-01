import os
import json
import time
import sys
import enum

from tqdm import tqdm
from pynvml import *
import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader

import train.utils as utils
from train.utils.subtype_enum import BinaryEnum
from train.patch_dataset import PatchDataset
import train.aim_logger as aim_logger
import train.aim_models as aim_models

# Folder permission mode
p_mode = 0o777
oldmask = os.umask(000)
nvmlInit()

class ChunkLookupException(Exception):
    pass

def build_model(model_config):
    return aim_models.DeepModel(model_config)

def setup_log_file(log_folder_path, log_name):
    os.makedirs(log_folder_path, exist_ok = True)
    l_path = os.path.join(log_folder_path, "log_{}.txt".format(log_name))
    sys.stdout = aim_logger.Logger(l_path)

def gpu_selector(gpu_to_use=-1):
    gpu_to_use = -1 if gpu_to_use == None else gpu_to_use
    deviceCount = nvmlDeviceGetCount()
    if gpu_to_use < 0:
        print("Auto selecting GPU") 
        gpu_free_mem = 0
        for i in range(deviceCount):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_usage = nvmlDeviceGetMemoryInfo(handle)
            if gpu_free_mem < mem_usage.free:
                gpu_to_use = i
                gpu_free_mem = mem_usage.free
            print("GPU: {} \t Free Memory: {}".format(i, mem_usage.free))
    print("Using GPU {}".format(gpu_to_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_to_use)
    return gpu_to_use

class ModelTrainer(object):

    @classmethod
    def create_patch_pattern(cls, patch_pattern):
        if type(patch_pattern) is str:
            patch_pattern = patch_pattern.split('/')
        return {k: i for i,k in enumerate(patch_pattern)}

    @classmethod
    def create_category_enum(cls, is_binary, subtypes=None):
        '''Create CategoryEnum

        Parameters
        ----------
        is_binary : bool
        subtypes : None or list

        Returns
        -------
        enum.Enum
        '''
        if is_binary:
            return BinaryEnum
        else:
            if subtypes:
                return enum.Enum('SubtypeEnum', subtypes)
            else:
                raise NotImplementedError('create_category_enum: is_binary is True and no subtypes given')

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
        self.CategoryEnum = self.create_category_enum(
                self.is_binary, subtypes)
        self.patch_pattern = self.create_patch_pattern(patch_pattern)
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

    @classmethod
    def from_config_file(cls, config_file_location):
        pass

    @classmethod
    def from_arguments(cls, config_file_location):
        pass

    def load_model_config(self):
        with open(self.model_config_location) as f:
            return json.load(f)

    def load_chunks(self, chunk_ids):
        """Load patch paths from specified chunks in chunk file

        Parameters
        ----------
        chunks : list of int
            The IDs of chunks to retrieve patch paths from

        Returns
        -------
        list of str
            Patch paths from the chunks
        """
        patch_paths = []
        with open(self.chunk_file_location) as f:
            data = json.load(f)
            chunks = data['chunks']
            for chunk in data['chunks']:
                if chunk['id'] in chunk_ids:
                    patch_paths.extend(chunk['imgs'])
        if len(patch_paths) == 0:
            raise ChunkLookupException(
                    f"chunks {tuple(chunk_ids)} not found in {self.chunk_file_location}")
        return patch_paths
        
    def extract_label_from_patch(self, patch_path):
        """Get the label value according to CategoryEnum from the patch path

        Parameters
        ----------
        patch_path : str

        Returns
        -------
        int
            The label id for the patch
        """
        '''
        Returns the CategoryEnum
        '''
        patch_id = utils.create_patch_id(patch_path, self.patch_pattern)
        label = utils.get_label_by_patch_id(patch_id, self.patch_pattern,
                self.CategoryEnum, is_binary=self.is_binary)
        return label.value

    def extract_labels(self, patch_paths):
        return list(map(self.extract_label_from_patch, patch_paths))

    def create_data_loader(self, chunk_ids, color_jitter=False, shuffle=False):
        patch_paths = self.load_chunks(chunk_ids)
        labels = self.extract_labels(patch_paths)
        patch_dataset = PatchDataset(patch_paths, labels, color_jitter=color_jitter)
        return DataLoader(patch_dataset, batch_size=self.batch_size, 
                shuffle=shuffle, num_workers=self.num_patch_workers)

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
            print(f'Epoch: {epoch}')
            print(f'Peak accuracy: {max_val_acc}')
            print(f'Peach accuracy at iteration: {max_val_acc_idx}')

    def run(self):
        if self.log_dir_location:
            setup_log_file(self.log_dir_location, self.train_instance_name)
        print(f'Instance name: {self.train_instance_name}')
        gpu_selector(self.gpu_id)
        training_loader = self.create_data_loader(self.training_chunks,
                color_jitter=True, shuffle=self.training_shuffle)
        validation_loader = self.create_data_loader(self.validation_chunks,
                shuffle=self.validation_shuffle)
        model = build_model(self.model_config)
        self.train(model, training_loader, validation_loader)
