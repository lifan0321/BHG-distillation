import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from datasets_recon.GeneralDataset import TaskData

class SimpleDataModule():
    def __init__(self,
                 nii_path,
                 fold_path,
                 fold_index,
                 batch_size,
                 num_workers,
                 aug,
                 input_D,
                 input_H,
                 input_W,              
                 seed: int = 20230329,
                 pin_memory: bool = False,
                 ):
        self.nii_path = nii_path
        # dataset info
        self.fold_path = fold_path

        # training configures
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.pin_memory = pin_memory


        self.ds_train = TaskData(nii_path, fold_path, 'train', aug, input_D, input_H, input_W, fold_index)
        self.ds_test = TaskData(nii_path, fold_path, 'val', aug, input_D, input_H, input_W, fold_index)
        self.ds_test = TaskData(nii_path, fold_path, 'test', aug, input_D, input_H, input_W, fold_index)

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)

        sampler = WeightedRandomSampler(self.ds_train.get_sample_weights(), len(self.ds_train.get_sample_weights()),
                                        generator=generator)
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            return DataLoader(self.ds_val, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=1, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A test test set was not initialized.")

