from __future__ import print_function, division
import numpy as np
from torch.utils.data import Dataset
import random
import csv
import nibabel as nib
import os
import torch
import pandas as pd
from monai.transforms import (Compose, ScaleIntensity, CenterSpatialCrop,
                              RandScaleIntensityd, RandRotated, RandZoomd)


class TaskData(Dataset):
    """
    this class will load data for a specific task, thus if the label of the task is missing for a case,
    that case will be omitted by the dataloader
    """

    def __init__(self,
                 nii_path,
                 fold_path,
                 stage,  # stage could be 'train' or 'valid' or 'test'
                 aug,
                 fold_index = 1,
                 seed=20230329,  # random seed
                 ):

        random.seed(seed)
        self.nii_path = os.path.join(nii_path, "Brain")
        self.fold_path = os.path.join(fold_path, str(fold_index), stage)
        self.stage = stage
        self.trans = Compose([ScaleIntensity(), CenterSpatialCrop([160, 192, 160])])
        self.trans_aug_d = Compose([RandScaleIntensityd(prob=0.5, factors=0.05, keys=['PET']),
                                    RandRotated(prob=0.5, range_x=0.05, range_y=0.05, range_z=0.05, keys=['PET']),
                                    RandZoomd(prob=0.2, min_zoom=0.9, max_zoom=1.1, keys=['PET'])])

        self.data_list = self.load_samples()
        self.aug = aug

    def load_samples(self):
        sample_list = []
        for class_folder in os.listdir(self.fold_path):
            class_folder_path = os.path.join(self.fold_path, class_folder)
            for subnamedir in os.listdir(class_folder_path):
                feats_brain_path = os.path.join(self.nii_path, subnamedir + ".nii.gz")

                sample_list.append((feats_brain_path, int(class_folder)))

        return sample_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Read Info From Dataframe
        pet_filepath, label = self.data_list[idx]
        label_tensor = torch.tensor(label)
        pet_data = nib.load(pet_filepath).get_fdata()

        pet_data = torch.from_numpy(np.expand_dims(pet_data, axis=0)).type(torch.float32)
        pet_data = self.trans(pet_data)
        if self.aug and self.stage == 'train':
            data = {'PET': pet_data}
            data = self.trans_aug_d(data)
            pet_data = data['PET']
        
        return pet_data, label_tensor

    def get_sample_weights(self):
        class_counts = {}

        for _, label in self.data_list:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        weights = [1.0 / class_counts[label] for _, label in self.data_list]
        return weights
    

