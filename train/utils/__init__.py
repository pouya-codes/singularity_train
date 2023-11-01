from train.utils.subtype_enum import BinaryEnum
from pathlib import Path
# from pynvml import *
# from email.mime.text import MIMEText
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import cohen_kappa_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
# from sklearn.metrics import roc_auc_score
# from sklearn.metrics import confusion_matrix
# import numpy as np
# import matplotlib.path as pltPath
# import subprocess
# import smtplib
# import socket
# import glob
# import csv
import re
# import os
# import random
# import torch
# import copy
# import collections
# import h5py
# import json

DATASET_TO_PATIENT_REGEX = {
    'ovcare': re.compile(r"^[A-Z]*-?(\d*).*\(?.*\)?.*$"),
    'tcga':  re.compile(r"^(TCGA-\w+-\w+)-")
}

DATASET_ORIGINS = DATASET_TO_PATIENT_REGEX.keys()

def get_patient_regex(dataset_origin):
    return DATASET_TO_PATIENT_REGEX[dataset_origin.lower()]

def strip_extension(path):
    """Function to strip file extension

    Parameters
    ----------
    path : string
        Absoluate path to a slide

    Returns
    -------
    path : string
        Path to a file without file extension
    """
    p = Path(path)
    return str(p.with_suffix(''))

def create_patch_id(path, patch_pattern):
    """Function to create patch id

    Parameters
    ----------
    path : string
        Absolute path to a patch

    patch_pattern : dict

    Returns
    -------
    patch_id : string
        Remove useless information before patch id for h5 file storage
    """
    len_of_patch_id = -(len(patch_pattern) + 1)
    patch_id = strip_extension(path).split('/')[len_of_patch_id:]
    patch_id = '/'.join(patch_id)
    return patch_id

def get_slide_by_patch_id(patch_id, patch_pattern):
    """Function to obtain slide id from patch id

    Parameters
    ----------
    patch_id : string
    patch_pattern : dict

    Returns
    -------
    slide_id : string
        Slide id extracted from `patch_id`
    """
    slide_id = patch_id.split('/')[patch_pattern['slide']]
    return slide_id


def get_label_by_patch_id(patch_id, patch_pattern, CategoryEnum, is_binary=False):
    """Function to obtain label from patch id

    Parameters
    ----------
    patch_id : string
    patch_pattern : dict
    CategoryEnum : enum.Enum
    is_binary : bool
        For binary classification, i.e., we will use BinaryEnum instead of SubtypeEnum

    Returns
    -------
    label: enum.Enum
        label from CategoryEnum

    """
    label = patch_id.split('/')[patch_pattern['annotation' if is_binary else 'subtype']]
    return CategoryEnum[label if is_binary else label.upper()]


def get_patient_by_slide_id(slide_id, dataset_origin='ovcare'):
    """
    Parameters
    ----------
    slide_id : str
        The slide ID i.e.
        For OVCARE the slide ID is VOA-1011A and patient ID is VOA-1011
        For TCGA the slide ID is TCGA-A5-A0GH-01Z-00-DX1.22005F4A-0E77-4FCB-B57A-9944866263AE
            and the patient ID is TCGA-A5-A0GH

    dataset_origin : str
        The dataset origin that determines the regex for the patient ID

    Returns
    -------
    str
        The patient ID from the slide ID
    """
    match = re.search(get_patient_regex(dataset_origin), slide_id)
    if match:
        return match.group(1)
    else:
        raise NotImplementedError(
            '{} is not detected by get_patient_regex(dataset_origins)'.format(slide_id))


def create_subtype_patient_slide_patch_dict(patch_paths, patch_pattern, CategoryEnum,
        is_binary=False, dataset_origin='ovcare'):
    """Function to patch ids sorted by {subtype: {patient: {slide_id: [patch_id]}}

    Parameters
    ----------
    patch_paths : list
        List of absolute patch paths
    
    patch_pattern : dict
        Dictionary describing the directory structure of the patch paths. The words are 'annotation', 'subtype', 'slide', 'magnification'

    CategoryEnum : enum.Enum
        The enum representing the categories and is one of (SubtypeEnum, BinaryEnum)

    is_binary : bool
        Whether we want to categorize patches by the Tumor/Normal category (true) or by the subtype category (false)

    dataset_origin : str
        The origins of the slide dataset the patches are generated from. One of DATASET_ORIGINS

    Returns
    -------
    subtype_patient_slide_patch : dict
        {subtype: {patient: {slide_id: [patch_id]}}

    """
    subtype_patient_slide_patch = {}
    for patch_path in patch_paths:
        patch_id = create_patch_id(patch_path, patch_pattern)
        patch_subtype = get_label_by_patch_id(patch_id, patch_pattern, CategoryEnum,
                is_binary=is_binary).name
        if patch_subtype not in subtype_patient_slide_patch:
            subtype_patient_slide_patch[patch_subtype] = {}
        slide_id = get_slide_by_patch_id(patch_id, patch_pattern)
        patient_id = get_patient_by_slide_id(slide_id, dataset_origin=dataset_origin)
        if patient_id not in subtype_patient_slide_patch[patch_subtype]:
            subtype_patient_slide_patch[patch_subtype][patient_id] = {}
        if slide_id not in subtype_patient_slide_patch[patch_subtype][patient_id]:
            subtype_patient_slide_patch[patch_subtype][patient_id][slide_id] = []
        subtype_patient_slide_patch[patch_subtype][patient_id][slide_id] += [patch_path]
    return subtype_patient_slide_patch

def patient_slide_patch_count(patch_ids_path, prefix, is_multiscale):
    '''
    TODO: make use of this
    '''
    # subtype_names = list(BinaryEnum.__members__.keys())
    subtype_names = [s.name for s in SubtypeEnum]
    patch_ids = read_data_ids(patch_ids_path)
    patch_dict = dict(zip(subtype_names, [0 for s in subtype_names]))
    slide_dict = dict(zip(subtype_names, [set() for s in subtype_names]))
    patient_dict = dict(zip(subtype_names, [set() for s in subtype_names]))

    for patch_id in patch_ids:
        patch_info = patch_id.split('/')
        label = patch_info[0]
        slide_id = get_slide_by_patch_id(patch_id, is_multiscale=is_multiscale)
        patient_id = get_patient_by_patch_id(
            patch_id, is_multiscale=is_multiscale)
        patch_dict[label] += 1
        slide_dict[label].add(slide_id)
        patient_dict[label].add(patient_id)

    def _latex_formatter(counts, prefix, percentage=False):
        if not percentage:
            print(r'{} & \num[group-separator={{,}}]{{{}}} & \num[group-separator={{,}}]{{{}}} & \num[group-separator={{,}}]{{{}}} & \num[group-separator={{,}}]{{{}}} & \num[group-separator={{,}}]{{{}}} & \num[group-separator={{,}}]{{{}}} \\'.format(
                prefix, int(counts[0]), int(counts[1]), int(counts[2]), int(counts[3]), int(counts[4]), int(counts.sum())))
        else:
            print(r'{} & {}\% & {}\% & {}\% & {}\% & {}\% & \num[group-separator={{,}}]{{{}}} \\'.format(
                prefix, np.around(counts[0]/counts.sum() * 100, decimals=2), np.around(counts[1]/counts.sum() * 100, decimals=2), np.around(counts[2]/counts.sum() * 100, decimals=2), np.around(counts[3]/counts.sum() * 100, decimals=2), np.around(counts[4]/counts.sum() * 100, decimals=2), int(counts.sum())))

    slide_count = np.zeros(5)
    patient_count = np.zeros(5)
    patch_count = np.zeros(5)
    total_patients = set()
    total_slides = set()
    for idx, subtype_name in enumerate(subtype_names):
        slide_count[idx] = len(slide_dict[subtype_name])
        patient_count[idx] = len(patient_dict[subtype_name])
        patch_count[idx] = patch_dict[subtype_name]
        total_patients = total_patients.union(patient_dict[subtype_name])
        total_slides = total_slides.union(slide_dict[subtype_name])

    _latex_formatter(patient_count, 'Patient in ' + prefix)
    _latex_formatter(slide_count, 'Slide in ' + prefix)
    _latex_formatter(patch_count, 'Patch in ' + prefix, percentage=True)
    print('Total Patients: {}'.format(len(total_patients)))
    print('Total Slides: {}'.format(len(total_slides)))

    return patient_dict
