import argparse
import os
from utils import mkdirs
from datetime import datetime


class Option:
    """This class defines options used during both training and CNN_PET_ADCN time. It also implements several helper
    functions such as parsing, printing, and saving the options. It also gathers additional options defined in
    <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initialized"""
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.opt = None

    def initialize(self, parser):
        """Define the common options that are used in both training and CNN_PET_ADCN."""
        # basic settings
        parser.add_argument('--organ', default="brain", type=str)
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints')
        parser.add_argument("--fold_index", default=0, type=int, help="current fold_index")
        parser.add_argument("--feats_path", default="./", type=str, help="dataset directory")
        parser.add_argument("--fold_path", default="./", type=str, help="dataset pdf directory")
        parser.add_argument('--output', type=str, default='./output', help='Output folder for both tensorboard and the best model')
        parser.add_argument("--num_workers", default=4, type=int, help="number of workers")
        parser.add_argument("--batch_size", default=2, type=int, help="number of batch size")
        parser.add_argument("--dropout", default="0.5", type=float, help="dropout for network")

        parser.add_argument("--num_classes", default=0, type=int, help="num_classes")
        parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
        parser.set_defaults(no_cuda=False)

        parser.add_argument("--pretrain_path_brain", default="xxx", type=str, help="pretrain path b")
        parser.add_argument("--pretrain_path_heart", default="xxx", type=str, help="pretrain path h")
        parser.add_argument("--pretrain_path_gut", default="xxx", type=str, help="pretrain path g")
        parser.add_argument("--model_depth", default=10, type=int, help="model layer")
        parser.add_argument("--resnet_shortcut", default="A", type=str, help="short cut type")
        parser.add_argument("--input_D", default=14, type=int, help="input_D")
        parser.add_argument("--input_H", default=28, type=int, help="input_H")
        parser.add_argument("--input_W", default=28, type=int, help="input_W")
        parser.add_argument("--aug", default=False, type=bool, help="augmentaion")

        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=1e-3)

        parser.add_argument('--sample', type=bool, default=True)
        parser.add_argument('--freeze_layers', type=str, default="[]", help="List of layers")

        return parser

    def print_options(self, opt):
        """Print and save options
        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        print(message)

        # save to the disk
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        expr_dir = os.path.join(opt.checkpoints_dir, f'{opt.organ}_{opt.model_depth}_{current_time}')
        mkdirs(expr_dir)
        checkpoint_dir = os.path.join(expr_dir, str(opt.fold_index))
        mkdirs(checkpoint_dir)


        opt.expr_dir = expr_dir

        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        print(f'Create opt file opt.txt')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        self.parser = self.initialize(self.parser)
        self.opt = self.parser.parse_args()
        self.print_options(self.opt)
        return self.opt
