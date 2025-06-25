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
from model import generate_model
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
from datasets import SimpleDataModule
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

    dm = SimpleDataModule(opt.feats_path, opt.fold_path, 2 * opt.fold_index + 1, opt.batch_size, opt.num_workers, opt.aug, seed)
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    opt.model = "resnet"
    resNet = generate_model(opt)
    # params = [{'params': parameters}]
    # print (resNet)
    optimizer = torch.optim.SGD(params=resNet.parameters(), lr=1e-03, momentum=0.9, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    # optims = torch.optim.AdamW(params=resNet.parameters(), lr=1e-04, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    post_pred = Compose([Activations(softmax=True)])
    post_label = Compose([AsDiscrete(to_onehot=2)])
    AUC = ROCAUCMetric(average='macro')
    AUC_val = ROCAUCMetric(average='macro')
    AUC_test = ROCAUCMetric(average='macro')

    loss_epoch = CumulativeAverage()
    best_auc = 0.0
    best_auc_f1 = 0.0
    best_f1 = 0.0

    start_time = time.time()

    for epoch in range(opt.epochs): #200
        # print("epoch: ", epoch)
        resNet.train()
        label_pred = []
        label_real = []   
        loss_epoch.reset()
        AUC.reset()
        
        for i, (inputs, labels) in enumerate(train_dataloader):
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = resNet(inputs)

            criterion = nn.CrossEntropyLoss()
            loss = criterion(outputs, labels)

            loss.backward()

            loss_epoch.append(loss)
            optimizer.step()
            # StepLR.step()
            label_real += [i for i in decollate_batch(labels)]
            label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]
            AUC(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        elapsed = time.time() - start_time
        elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
        logger_main.print_message(f'Epoch {epoch + 1}/{opt.epochs} - Elapsed time {elapsed}')

        scheduler.step()
        loss_results = loss_epoch.aggregate()
        cm_train = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger_main.print_message(f"Epoch :{int(epoch)} ")
        logger_main.print_message(f"Trainng    - Loss:{float(loss_results):.4f} "
                             f"ACC:{float(cm_train.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_train.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_train.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_train.F1.values())[1]):.4f} "
                             f"AUC:{AUC.aggregate():.4f}")

        resNet.eval()  
        loss_epoch.reset()
        AUC_val.reset()
        label_pred = []
        label_real = []                  
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(val_dataloader):
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
                outputs = resNet(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss_epoch.append(loss)

                label_real += [i for i in decollate_batch(labels)]
                label_pred += [post_pred(i).detach().cpu().numpy().argmax() for i in decollate_batch(outputs)]
                AUC_val(y_pred=[post_pred(i) for i in decollate_batch(outputs)],
                    y=[post_label(i) for i in decollate_batch(labels, detach=False)])

        loss_results = loss_epoch.aggregate()
        cm_val = ConfusionMatrix(actual_vector=label_real, predict_vector=label_pred)
        logger_main.print_message(f"Validation    - Loss:{float(loss_results):.4f} "
                             f"ACC:{float(cm_val.Overall_ACC):.4f} "
                             f"SEN:{float(list(cm_val.TPR.values())[1]):.4f} "
                             f"SPE:{float(list(cm_val.TNR.values())[1]):.4f} "
                             f"F1:{float(list(cm_val.F1.values())[1]):.4f} "
                             f"AUC:{AUC_val.aggregate():.4f}"
                             )

        val_auc_value = float(AUC_test.aggregate())
        f1 = float(list(cm_val.F1.values())[1])


        if val_auc_value >= best_auc:
            print(f'Model saved !!!')
            best_auc = val_auc_value
            logger_test.print_message(f"Epoch best AUC:{int(epoch)} ")
            logger_test.print_message(f"ACC:{float(cm_val.Overall_ACC):.4f} "
                                f"SEN:{float(list(cm_val.TPR.values())[1]):.4f} "
                                f"SPE:{float(list(cm_val.TNR.values())[1]):.4f} "
                                f"F1:{float(list(cm_val.F1.values())[1]):.4f} "
                                f"AUC:{AUC_test.aggregate():.4f}"
                                )
            save_name_epoch = os.path.join(checkpoint_dir, "epoch_best_auc" + str(opt.fold_index) + ".ckpt")
            outdict = resNet.state_dict()
            torch.save(outdict, save_name_epoch)
          

if __name__ == "__main__":
    main()