import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import numpy as np
from munch import Munch
from monai.utils import set_determinism
# from tqdm import tqdm
from model_recon import generate_model
from torch import optim
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
from monai.data import decollate_batch
from pycm import ConfusionMatrix
from monai.metrics import CumulativeAverage, ROCAUCMetric
from monai import transforms
from monai.transforms import Compose, Activations, AsDiscrete
from torchvision import transforms
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from options import Option
from utils import Logger
from datasets_recon import SimpleDataModule
import datetime
import time

def plot(data, dir, organ_name_type):
    try:
        x = list(range(len(data)))
        plt.plot(x, data, label=organ_name_type)
        plt.xlabel('epoch')
        plt.ylabel(organ_name_type)
        plt.title('Eval ' + organ_name_type)
        plt.xlim(0, len(data))
        plt.legend(loc='best')
        plt.savefig(dir + '/' + organ_name_type  + '.png')
        plt.close()
    except Exception as e:
        print(e)

def seed_torch(seed):
    random.seed(seed)
    np.random.seed(seed)
    set_determinism(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def save_metadata(metadata, output_path, output_postfix=''):
    """
    save the metadata of the dataset
    :return:
    """
    save = Compose([transforms.SaveImage(output_dir=output_path, output_postfix=output_postfix, print_log=True, resample=False, separate_folder=False), ])
    save(metadata)


def main():
    opt = Option().parse()
    save_dir = opt.expr_dir
    checkpoint_dir = os.path.join(save_dir, str(opt.fold_index))
    

    logger_main = Logger(checkpoint_dir)
    logger_test = Logger(save_dir)
    logger_test.print_message(f'************Fold {str(opt.fold_index)}************')

    print(f'Successfully load datasets..... in model - resNet{opt.model_depth}')

    seed = 20230329
    seed_torch(seed)
    print(f'The random seed is {seed}')

    dm = SimpleDataModule(opt.feats_path, opt.fold_path, 2 * opt.fold_index + 1, opt.batch_size, opt.num_workers, opt.aug, opt.input_D, opt.input_H, opt.input_W, seed)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    opt.model = "resnet"
    resNet = generate_model(opt)

    optimizer = torch.optim.SGD(params=resNet.parameters(), lr=1e-03, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_epoch = CumulativeAverage()
    best_loss = 100000000

    start_time = time.time()

    criterion = nn.L1Loss().to('cuda')
    for epoch in range(opt.epochs):
        resNet.train()
        loss_epoch.reset()
        
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = resNet(inputs)
            loss = criterion(outputs, labels)

            loss.backward()

            loss_epoch.append(loss)
            optimizer.step()
        scheduler.step()

        loss_results = loss_epoch.aggregate()
        logger_main.print_message(f"Epoch :{int(epoch)} ")
        logger_main.print_message(f"Trainng    - Loss:{float(loss_results):.4f} ")   

        resNet.eval()  
        loss_epoch.reset()

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_dataloader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = resNet(inputs)

                loss = criterion(outputs, labels)
                loss_epoch.append(loss)

        loss_results = loss_epoch.aggregate()
        logger_main.print_message(f"Test    - Loss:{float(loss_results):.4f} ")    

        if loss_results <= best_loss:
            print(f'Model saved !!!')
            best_loss = loss_results
            logger_test.print_message(f"Epoch best AUC:{int(epoch)} ")
            logger_test.print_message(f"Test    - Loss:{float(loss_results):.4f} ")    

            save_name_epoch = os.path.join(checkpoint_dir, "epoch_best_loss" + str(opt.fold_index) + ".ckpt")
            outdict = resNet.state_dict()
            torch.save(outdict, save_name_epoch)

          

if __name__ == "__main__":
    main()