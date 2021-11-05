#!/usr/bin/env python

"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

To train the model, this script must be run in the command prompt. The command-line parsing module "argparse" is used
to take a number of inputs from the user such as the architecture type of the deep learning model, hyper-parameters,
path to the data and etc. You may find all arguments in the "if block" at the last part of the script.
"""

# import necessary modules
import argparse
import os
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
import random
import torch
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# To have reproducible results the random seed is set to 42.
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(config):
    if config.model_type not in ['TernausNet16']:
        print(
            'ERROR!! model_type should be selected in TernausNet16')
        print('Your input for model_type was %s' % config.model_type)
        return

    # Create directories for results if not exist
    if not os.path.exists(config.model_path):
        os.makedirs(config.model_path)
    if not os.path.exists(config.result_path):
        os.makedirs(config.result_path)
    # config.result_path = os.path.join(config.result_path, config.model_type)
    # if not os.path.exists(config.result_path):
    # os.makedirs(config.result_path)
    print(config)

    # Load training data
    train_loader = get_loader(image_path=config.train_path, image_size=config.image_size, batch_size=config.batch_size,
                              num_workers=config.num_workers, mode='train', augmentation_prob=config.augmentation_prob,
                              shuffle_flag=True, pretrained=bool(config.pretrained))
    # Load validation data
    valid_loader = get_loader(image_path=config.valid_path, image_size=config.image_size, batch_size=config.batch_size,
                              num_workers=config.num_workers, mode='valid',
                              augmentation_prob=0., shuffle_flag=False, pretrained=bool(config.pretrained))
    # Load test data
    test_loader = get_loader(image_path=config.test_path, image_size=config.image_size, batch_size=1,
                             num_workers=0, mode='valid', augmentation_prob=0., shuffle_flag=False,
                             pretrained=bool(config.pretrained))

    # Define a solver instance
    solver = Solver(config, train_loader, valid_loader, test_loader)
    solver.train()
    solver.pred()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256)  # input size of the model (width = height)
    parser.add_argument('--img_ch', type=int, default=1, help='number of channels of the input data')
    parser.add_argument('--output_ch', type=int, default=1, help='number of channels of the output data')
    parser.add_argument('--pretrained', type=int, default=1, help='to use pre-trained weights input must be 1')
    parser.add_argument('--num_epochs', type=int, default=100, help='number epochs for training')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help=' momentum1 in the Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='momentum2 in the Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay')
    parser.add_argument('--lossfunc', type=str, default='DiceLoss', help='DiceLoss')
    parser.add_argument('--augmentation_prob', type=float, default=0.5)
    parser.add_argument('--number_layers_freeze', type=int, default=0)
    parser.add_argument('--model_type', type=str, default='TernausNet16', help='TernausNet16')
    parser.add_argument('--model_path', type=str, default='../models', help='path to save the best model')
    parser.add_argument('--train_path', type=str, default='../dataset/DIC_crack_dataset/train/', help='path to training images')
    parser.add_argument('--valid_path', type=str, default='../dataset/DIC_crack_dataset/valid/', help='path to validation images')
    parser.add_argument('--test_path', type=str, default='../dataset/DIC_crack_dataset/test/', help='path to test images')
    parser.add_argument('--result_path', type=str, default='../results/', help='path to save results')
    parser.add_argument('--cuda_idx', type=int, default=1, help='if cuda available = 1 otherwise = 0')
    config = parser.parse_args()
    main(config)
