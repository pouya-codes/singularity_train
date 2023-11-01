import os
import json
import time
import sys
import enum
import csv

import yaml, random
from submodule_utils.accuracy.slide_level_accuracy import SlideLevelAccuracy
from submodule_utils.metadata.heatmaps import generate_heatmaps
from submodule_utils.metadata.gradcamH5 import GradCAM_AIM
from tqdm import tqdm
from pynvml import *
import numpy as np
import torch, h5py
import torchvision
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

import submodule_utils as utils
from submodule_cv import (ChunkLookupException, setup_log_file,
                                                  gpu_selector, PatchHanger, EarlyStopping, mixup_data)


class ModelTrainer(PatchHanger):
        """Trains a model

        Attributes
        ----------
        experiment_name : str
                Experiment name

        instance_name : str
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
                self.instance_name = f'{self.experiment_name}_{timestamp}'

                # hyperparameters
                self.train_model = config.train_model
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
                self.save_model_for_export = config.save_model_for_export
                self.model_config_location = config.model_config_location
                self.model_config = self.load_model_config()
                # optional
                self.num_patch_workers = config.num_patch_workers
                self.num_validation_batches = config.num_validation_batches
                self.gpu_id = config.gpu_id
                self.number_of_gpus = config.number_of_gpus
                self.training_shuffle = config.training_shuffle
                self.validation_shuffle = config.validation_shuffle

                # early stopping parameters
                self.early_stopping = not isinstance(config.subparser, type(None)) and config.use_early_stopping
                if self.early_stopping:
                        self.patience = config.patience
                        self.delta = config.delta

                # testing model parameters
                self.testing_model = config.testing_model
                self.test_model_file_location = config.test_model_file_location
                self.old_version = config.old_version
                self.test_log_dir_location = config.test_log_dir_location
                self.detailed_test_result = config.detailed_test_result
                self.testing_shuffle = config.testing_shuffle
                self.test_chunks = config.test_chunks
                self.seed = config.seed
                self.slide_level_accuracy = config.calculate_slide_level_accuracy
                self.slide_level_accuracy_threshold = config.slide_level_accuracy_threshold
                self.slide_level_accuracy_verbose = config.slide_level_accuracy_verbose

                self.generate_heatmaps = config.generate_heatmaps
                if self.generate_heatmaps:
                        self.heatmaps_dir_location = config.heatmaps_dir_location
                        self.slides_location = config.slides_location
                self.gradcam = config.gradcam
                self.gradcam_h5 = config.gradcam_h5
                if self.gradcam:
                        self.gradcam_dir_location = config.gradcam_dir_location
                        self.slides_location = config.slides_location
                        
                        

                self.best_model_state_dict = None

                if config.writer_log_dir_location:
                        self.writer_log_dir_location = config.writer_log_dir_location
                else:
                        self.writer_log_dir_location = os.path.join(
                                self.log_dir_location, self.experiment_name)
                if not os.path.exists(self.writer_log_dir_location):
                        os.makedirs(self.writer_log_dir_location)
                self.writer = SummaryWriter(log_dir=self.writer_log_dir_location)
                self.model_file_location = os.path.join(self.model_dir_location,
                                                                                                f'{self.instance_name}.pth')
                self.config = config

                #########
                # Amirali
                self.class_weight = self.model_config["use_weighted_loss"]["weight"] if \
                        self.model_config["use_weighted_loss"]["use_weighted_loss"] else None
                self.MixUp = True if 'mix_up' in self.model_config and self.model_config['mix_up']['use_mix_up'] else False
                self.scheduler_step = config.scheduler_step
                if "scheduler" in self.model_config:
                        if self.scheduler_step is None:
                                raise ValueError("scheduler_step is not determined!")
                        else:
                                self.scheduler = True
                                print("Scheduler is selected!")
                else:
                        self.scheduler = False
                # freeze training parameters
                self.freeze_training = not isinstance(config.subparser, type(None)) and config.use_freeze_training
                if self.freeze_training:
                        self.freeze_epochs = config.freeze_epochs
                        self.unfreeze_epochs = config.unfreeze_epochs
                        self.base_lr = config.base_lr
                        self.lr_mult = config.lr_mult
                        self.use_scheduler = config.use_scheduler
                        self.scheduler = True if self.use_scheduler else self.scheduler

                # representation learning parameters
                self.representation_learning = not isinstance(config.subparser,
                                                                                                          type(None)) and config.use_representation_learning
                if self.representation_learning:
                        self.model_path_location = config.model_path_location
                        self.train_feature_extractor = config.train_feature_extractor

                self.progressive_resizing = config.progressive_resizing


        def print_parameters(self):
                parameters = self.config.__dict__.copy()
                parameters['instance_name'] = self.instance_name
                parameters['model_file_location'] = self.model_file_location
                parameters['writer_log_dir_location'] = self.writer_log_dir_location
                payload = yaml.dump(parameters)
                print('---')  # begin YAML
                print(payload)
                print('...')  # end YAML

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
                                cur_data, cur_label, _, _ = data
                                cur_data = cur_data.cuda()
                                cur_label = cur_label.cuda()
                                logits, pred_prob, output = model.forward(cur_data)
                                val_loss += model.get_loss(logits, cur_label, output).item()

                                pred_labels += torch.argmax(pred_prob,
                                                                                        dim=1).cpu().numpy().tolist()
                                gt_labels += cur_label.cpu().numpy().tolist()
                                val_idx += 1

                model.model.train()
                return accuracy_score(gt_labels, pred_labels), (val_loss / val_idx)

        def compute_metric(self, labels, preds, probs, CategoryEnum, verbose=True, is_binary=False):
                """Function to compute the various metrics given predicted labels and ground truth labels

                Parameters
                ----------
                labels : numpy array
                        A row contains the ground truth labels
                preds: numpy array
                        A row contains the predicted labels
                probs: numpy array
                        A matrix and each row is the probability for the predicted patches or slides
                verbose : bool
                        Print detail of the computed metrics

                Returns
                -------
                overall_acc : float
                        Accuracy
                overall_kappa : float
                        Cohen's kappa
                overall_f1 : float
                        F1 score
                overall_auc : float
                        ROC AUC
                """
                print('\nComputing Metric:')
                overall_acc = accuracy_score(labels, preds)
                overall_kappa = cohen_kappa_score(labels, preds)
                overall_f1 = f1_score(labels, preds, average='macro')
                conf_mat = confusion_matrix(labels, preds).T
                acc_per_subtype = conf_mat.diagonal() / conf_mat.sum(axis=0) * 100
                acc_per_subtype[np.isinf(acc_per_subtype)] = 0.00
                if not is_binary and len(CategoryEnum) > 2:
                        try:
                                overall_auc = roc_auc_score(
                                        labels, probs, multi_class='ovr', average='macro')
                        except ValueError as e:
                                print('Warning:', e)
                                overall_auc = 0.00
                else:
                        overall_auc = roc_auc_score(labels, probs[:, 1], average='macro')
                # disply results
                if verbose:
                        print('Acc: {:.2f}\%'.format(overall_acc * 100))
                        print('Kappa: {:.4f}'.format(overall_kappa))
                        print('F1: {:.4f}'.format(overall_f1))
                        print('AUC ROC: {:.4f}'.format(overall_auc))
                        print('Confusion Matrix')
                        if is_binary:
                                print('||X||Actual Non-tumor||Actual Tumor||')
                                print('|Predicted Non-tumor|{}|{}|'.format(conf_mat[0][0], conf_mat[0][1]))
                                print('|Predicted Tumor|{}|{}|'.format(conf_mat[1][0], conf_mat[1][1]))
                        else:
                                output = '||X||'
                                for s in CategoryEnum:
                                        output += f'Actual {s.name}||'
                                output += '\n'
                                for i, s1 in enumerate(CategoryEnum):
                                        output += f'|Predicted {s1.name}|'
                                        for s2 in CategoryEnum:
                                                output += f'{conf_mat[s1.value][s2.value]}|'
                                        if i != len(CategoryEnum):
                                                output += '\n'
                                print(output)
                        print(repr(conf_mat))
                        # latex format
                        if is_binary:
                                print(
                                        '||Dataset||Non-tumor Accuracy||Tumor Accuracy||Weighted Accuracy||Kappa||F1 Score||AUC||Average '
                                        'Accuracy||')
                                print('|X|{:.2f}%|{:.2f}%|{:.2f}%|{:.4f}|{:.4f}|{:.4f}|{:.2f}%|'.format(
                                        acc_per_subtype[0], acc_per_subtype[1], overall_acc * 100, overall_kappa, overall_f1, overall_auc,
                                        acc_per_subtype.mean()))
                        else:
                                output = '||Dataset||'
                                for s in CategoryEnum:
                                        output += f'{s.name} Accuracy||'
                                output += 'Weighted Accuracy||Kappa||F1 Score||AUC||Average Accuracy||\n|X|'
                                for s in CategoryEnum:
                                        output += '{:.2f}%|'.format(acc_per_subtype[s.value])
                                output += '{:.2f}%|{:.4f}|{:.4f}|{:.4f}|{:.2f}%|'.format(overall_acc * 100, overall_kappa, overall_f1,
                                                                                                                                                 overall_auc, acc_per_subtype.mean())
                                print(output)

                return overall_acc, overall_kappa, overall_f1, overall_auc

        def test(self, model, test_loader, tag):

                if (self.detailed_test_result):
                        detailed_output_file = open(os.path.join(self.test_log_dir_location, f'details_{self.instance_name}.csv'),
                                                                                'w')
                        detailed_output_writer = csv.writer(detailed_output_file, delimiter=',', quotechar='"',
                                                                                                quoting=csv.QUOTE_MINIMAL)
                        detailed_output_writer.writerow(["path", "predicted_label", "target_label", "probability", "chunk"])

                if (self.gradcam):
                    self.gradcam_instance = GradCAM_AIM(self.slides_location, self.CategoryEnum, self.patch_pattern,
                                                        self.gradcam_dir_location, model.model, self.gradcam_h5)
                

                pred_labels = []
                gt_labels = []
                # pred_probs = np.array([]).reshape(
                #                0, len(self.CategoryEnum)) if not self.is_binary else np.array([])
                pred_probs = np.array([]).reshape(
                        0, len(self.CategoryEnum))

                model.model.eval()
                # with torch.no_grad():
                prefix = 'Testing: '
                counter = 0 
                for data in tqdm(test_loader, desc=prefix,
                                                 dynamic_ncols=True, leave=True, position=0):

                        counter += 1
                        cur_data, cur_label, cur_path, cur_chunk = data
                        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        cur_data = cur_data.to(device)
                        cur_label = cur_label.to(device)
                        _, pred_prob, _ = model.forward(cur_data)

                        pred_label = torch.argmax(pred_prob,dim=1).cpu().detach().numpy().tolist()
                        pred_labels += pred_label

                        gt_label = cur_label.cpu().detach().numpy().tolist()
                        gt_labels += gt_label

                        pred_prob = pred_prob.cpu().detach().numpy()
                        pred_probs = np.vstack((pred_probs, pred_prob))

                        
                        if (self.gradcam):
                                self.gradcam_instance.get_gradcam(cur_path, cur_data, gt_label)


                        
                        if (self.detailed_test_result):
                            for path_, pred_label_, true_label_, pred_prob_, cur_chunk_ in zip(cur_path, pred_label, gt_label,pred_prob,cur_chunk.cpu().numpy()):
                                        detailed_output_writer.writerow([path_, pred_label_, true_label_, pred_prob_, cur_chunk_])
                

                if self.detailed_test_result:
                        detailed_output_file.close()
                        if self.slide_level_accuracy:
                                detailed_results = open(os.path.join(self.test_log_dir_location, f'details_{self.instance_name}.csv'))
                                slide_level = SlideLevelAccuracy(csv.reader(detailed_results), self.patch_pattern, self.CategoryEnum,
                                                                                                 self.slide_level_accuracy_threshold, self.slide_level_accuracy_verbose)
                                slide_level.calculate_slide_level_accuracy()

                        if (self.generate_heatmaps):
                                generate_heatmaps(os.path.join(self.test_log_dir_location, f'details_{self.instance_name}.csv'),
                                                                  self.patch_pattern, self.CategoryEnum, self.slides_location,
                                                                  self.heatmaps_dir_location)
                
                if (self.gradcam_h5):
                        self.gradcam_instance.save_gradcam_h5()
                               
                print(f"{tag} Results:\n{40 * '*'}")
                self.compute_metric(gt_labels, pred_labels, pred_probs, self.CategoryEnum, verbose=True,
                                                        is_binary=self.is_binary)
                print(f"{40 * '*'}")

        def train(self, model, training_loader, validation_loader, best_val_acc=None):
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
                max_val_acc = float('-inf') if best_val_acc is None else best_val_acc
                max_val_acc_idx = -1
                intv_loss = 0
                pred_labels = []
                gt_labels = []
                if self.MixUp:
                        gt_labels_mixed = []
                # initialize the early_stopping object
                if (self.early_stopping):
                        early_stopping = EarlyStopping(patience=self.patience, delta=self.delta)
                self.validation_interval = len(training_loader) if self.validation_interval == -1 else self.validation_interval
                if best_val_acc is None:
                        val_acc, val_loss = self.validate(model, validation_loader, iter_idx)
                        max_val_acc = val_acc
                        print(f'\nBefore training, validation accuracy is {val_acc}, validation loss is {val_loss}!')
                ###################
                # train the model #
                ###################
                for epoch in range(self.epochs):
                        prefix = f'Training Epoch {epoch}: '
                        for data in tqdm(training_loader, desc=prefix,
                                                         dynamic_ncols=True, leave=True, position=0):
                                iter_idx += 1
                                batch_data, batch_labels, _, _ = data
                                batch_data = batch_data.cuda()
                                batch_labels = batch_labels.cuda()
                                if self.MixUp:
                                        batch_data, batch_labels_mixed, lam = mixup_data(batch_data, batch_labels)
                                logits, probs, output = model.forward(batch_data)
                                if self.MixUp:
                                        model.optimize_parameters(logits, batch_labels, output,
                                                                                          batch_labels_mixed, lam)
                                else:
                                        model.optimize_parameters(logits, batch_labels, output)
                                if self.scheduler and self.scheduler_step.upper() == "BATCH":
                                        model.scheduler_step()
                                intv_loss += model.get_current_errors()
                                pred_labels += torch.argmax(probs,
                                                                                        dim=1).cpu().numpy().tolist()
                                gt_labels += batch_labels.cpu().numpy().tolist()
                                if self.MixUp:
                                        gt_labels_mixed += batch_labels_mixed.cpu().numpy().tolist()
                                if iter_idx % self.validation_interval == self.validation_interval - 1:
                                        val_acc, val_loss = self.validate(model, validation_loader, iter_idx)
                                        if self.MixUp:
                                                train_acc = lam * accuracy_score(gt_labels, pred_labels) + \
                                                                        (1 - lam) * accuracy_score(gt_labels_mixed, pred_labels)
                                        else:
                                                train_acc = accuracy_score(gt_labels, pred_labels)
                                        self.writer.add_scalars(f"{self.instance_name}/loss",
                                                                                        {
                                                                                                'validation': val_loss,
                                                                                                'train': intv_loss / self.validation_interval
                                                                                        }, iter_idx)
                                        self.writer.add_scalars(f"{self.instance_name}/accuracy",
                                                                                        {
                                                                                                'validation': val_acc,
                                                                                                'train': train_acc
                                                                                        }, iter_idx)
                                        print(f'\nEpoch: {epoch}, Iteration: {iter_idx}')
                                        print(f'Training: accuracy is {train_acc}, loss is {intv_loss / self.validation_interval}')
                                        print(f'Validation: accuracy is {val_acc}, loss is {val_loss}')
                                        intv_loss = 0
                                        pred_labels = []
                                        gt_labels = []
                                        if self.MixUp:
                                                gt_labels_mixed = []
                                        if max_val_acc <= val_acc:
                                                max_val_acc = val_acc
                                                max_val_acc_idx = iter_idx
                                                model.save_state(self.model_dir_location,
                                                                                 self.instance_name,
                                                                                 iter_idx, epoch)
                                                if (self.save_model_for_export):
                                                        torch.save(model,
                                                                           os.path.join(self.model_dir_location, "export_", self.instance_name, ".pt"))
                                                # store best state of model for testing
                                                self.best_model_state_dict = deepcopy(model.model.state_dict())
                                        if (self.early_stopping):
                                                early_stopping(val_loss, model.model)
                                                if early_stopping.early_stop:
                                                        print(f'\nEarly stopping at Epoch: {epoch}')
                                                        print(f'Peak accuracy: {max_val_acc}')
                                                        print(f'Peak accuracy at iteration: {max_val_acc_idx}')
                                                        self.writer.close()
                                                        return
                                self.writer.flush()
                        if self.scheduler and self.scheduler_step.upper() == "EPOCH":
                                model.scheduler_step()
                        print(f'\nEpoch: {epoch}')
                        print(f'Peak accuracy: {max_val_acc}')
                        print(f'Peak accuracy at iteration: {max_val_acc_idx}')
                        self.writer.close()
                        if self.scheduler:
                                print(
                                        f"Learning rates are: Feature_extraction={model.get_current_lr(0)}, Classifier={model.get_current_lr(1)}")
                # If nothing is saved
                if max_val_acc_idx == -1:
                        self.best_model_state_dict = deepcopy(model.model.state_dict())
                        model.save_state(self.model_dir_location,
                                                         self.instance_name,
                                                         iter_idx, epoch)
                        print("Saved model is initialized one!")
                return max_val_acc

        def freeze_train(self, model, training_loader, validation_loader):
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
                # Freeze
                self.epochs = self.freeze_epochs
                print(f"\nFreezing model for {self.epochs} epochs ...")
                model.freeze()
                model.update_optimizer_schedular(self.base_lr, use_scheduler=self.use_scheduler,
                                                                                 epoch=self.epochs, batch_per_epoch=len(training_loader))
                best_val_acc = self.train(model, training_loader, validation_loader)

                # Unfreeze
                self.epochs = self.unfreeze_epochs
                self.base_lr /= 2
                print(f"\nUnFreezing model for {self.epochs} epochs ...")
                model.unfreeze()
                model.update_optimizer_schedular([self.base_lr / self.lr_mult, self.base_lr],
                                                                                 use_scheduler=self.use_scheduler, epoch=self.epochs,
                                                                                 batch_per_epoch=len(training_loader))
                self.train(model, training_loader, validation_loader, best_val_acc=best_val_acc)
                self.base_lr *= 2

        def repr_train(self, model, training_loader, validation_loader):
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
                model.load_state_repr(self.model_path_location)
                if not self.train_feature_extractor:
                        # Only train classifier
                        # model.freeze_all()
                        # model.make_classifier_layer_trainable()
                        # train classifier + bn of feature extractor
                        model.freeze()
                        model.update_optimizer_schedular(learning_rate=None, use_scheduler=False,
                                                                                         epoch=self.epochs)

                self.train(model, training_loader, validation_loader)

        def run(self):
                if self.train_model:
                        setup_log_file(self.log_dir_location, self.instance_name)
                if self.testing_model:
                        setup_log_file(self.test_log_dir_location, self.instance_name)
                self.print_parameters()
                # gpu_devices = gpu_selector(self.gpu_id, self.number_of_gpus)
                gpu_devices = None
                model = self.build_model(gpu_devices, class_weight=self.class_weight)
                if (self.train_model):
                        for size in self.progressive_resizing:
                                if size != -1:
                                        print(f"\nTraining model with SIZE = {size}")
                                training_loader = self.create_data_loader(self.training_chunks, shuffle=self.training_shuffle,
                                                                                                                  training_set=True, size=size)
                                validation_loader = self.create_data_loader(self.validation_chunks, shuffle=self.validation_shuffle,
                                                                                                                        size=size)
                                if self.freeze_training:
                                        self.freeze_train(model, training_loader, validation_loader)
                                elif self.representation_learning:
                                        self.repr_train(model, training_loader, validation_loader)
                                else:
                                        self.train(model, training_loader, validation_loader)
                        if size != -1:
                                validation_loader = self.create_data_loader(self.validation_chunks, shuffle=self.validation_shuffle)
                        if self.best_model_state_dict:
                                model.model.load_state_dict(self.best_model_state_dict)
                        self.test(model, validation_loader, 'Validation')
                if (self.testing_model):
                        test_loader = self.create_data_loader(self.test_chunks, shuffle=self.testing_shuffle)
                        if (self.best_model_state_dict and not self.test_model_file_location):
                                model.model.load_state_dict(self.best_model_state_dict)
                        else:
                                if self.old_version:
                                        model.load_state_old_version(self.test_model_file_location)
                                else:
                                        model.load_state(self.test_model_file_location)
                        self.test(model, test_loader, 'Test')
