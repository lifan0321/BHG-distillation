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
                 fold_index,
                 seed,  # random seed
                 ):

        random.seed(seed)
        self.nii_path_brain = os.path.join(nii_path, "Brain")
        self.nii_path_heart = os.path.join(nii_path, "Heart")
        self.nii_path_gut = os.path.join(nii_path, "Gut")
        self.fold_path = os.path.join(fold_path, str(fold_index), stage)
        self.stage = stage
        self.trans_b = Compose([ScaleIntensity(), CenterSpatialCrop([160, 192, 160])])
        self.trans_h = Compose([ScaleIntensity(), CenterSpatialCrop([144, 128, 128])])
        self.trans_g = Compose([ScaleIntensity(), CenterSpatialCrop([264, 160, 400])])

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
                feats_brain_path = os.path.join(self.nii_path_brain, subnamedir + ".nii.gz")
                feats_heart_path = os.path.join(self.nii_path_heart, subnamedir + ".nii.gz")
                feats_gut_path = os.path.join(self.nii_path_gut, subnamedir + ".nii.gz")

                sample_list.append((feats_brain_path, feats_heart_path, feats_gut_path, int(class_folder)))

        return sample_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # Read Info From Dataframe
        pet_filepath_brain, pet_filepath_heart, pet_filepath_gut, label = self.data_list[idx]
        label_tensor = torch.tensor(label)
        pet_data_brain = nib.load(pet_filepath_brain).get_fdata()
        pet_data_brain = torch.from_numpy(np.expand_dims(pet_data_brain, axis=0)).type(torch.float32)
        pet_data_brain = self.trans_b(pet_data_brain)

        pet_data_heart = nib.load(pet_filepath_heart).get_fdata()
        pet_data_heart = torch.from_numpy(np.expand_dims(pet_data_heart, axis=0)).type(torch.float32)
        pet_data_heart = self.trans_h(pet_data_heart)

        pet_data_gut = nib.load(pet_filepath_gut).get_fdata()
        pet_data_gut = torch.from_numpy(np.expand_dims(pet_data_gut, axis=0)).type(torch.float32)
        pet_data_gut = self.trans_g(pet_data_gut)

        if self.aug and self.stage == 'train':
            data = {'PET': pet_data_brain}
            data = self.trans_aug_d(data)
            pet_data_brain = data['PET']

        return pet_data_brain, pet_data_heart, pet_data_gut, label_tensor, pet_filepath_brain

    def get_sample_weights(self):
        class_counts = {}

        for _, _, _, label in self.data_list:
            if label in class_counts:
                class_counts[label] += 1
            else:
                class_counts[label] = 1

        weights = [1.0 / class_counts[label] for _, _, _, label in self.data_list]
        return weights
    

