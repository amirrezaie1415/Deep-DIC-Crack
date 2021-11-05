"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

This script creates the folder "dataset", which contains training and validation images
and the corresponding masks (ground truth). The test data was separately given and placed
in the folders "test" and "test_GT".
"""

# import necessary modules
import os
import argparse
import random
import shutil
from shutil import copyfile
from misc import printProgressBar

# To have reproducible results the random seed is set to 100.

random.seed(100)

def rm_mkdir(dir_path):
    """
    Check whether a path exists of not. If it does not exist, it creates the "dir_path".
    EXAMPLE:  dir_path = '../dataset/train/'. If in this path does not already exist,
    folders "dataset" and "train" will be created.
    """
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s' % dir_path)
    os.makedirs(dir_path)
    print('Create path - %s' % dir_path)


def main(config):
    """
    The main function. As the input for this function, the object "config" containing
    the input parameters (see parser arguments at the end of this script).
    """
    # check the existence of train and validation paths (both input and ground truth).
    rm_mkdir(config.train_path)
    rm_mkdir(config.train_GT_path)
    rm_mkdir(config.valid_path)
    rm_mkdir(config.valid_GT_path)

    filenames = os.listdir(config.origin_data_path)
    data_list = []  # list of image names.
    GT_list = []  # list of ground truth files.

    for filename in filenames:
        ext = os.path.splitext(filename)[-1]
        if ext == '.tif':  # input images has the extension ".tif",
            # while the ground truth images has the extension ".mask"
            filename = filename[:-len('.tif')]
            data_list.append(filename + '.tif')
            GT_list.append(filename + '_mask.png')

    # determine the number of training and validation sets.
    num_total = len(data_list)
    num_valid = int(config.valid_ratio * num_total)
    num_train = num_total - num_valid

    print('\nNum of train set : ', num_train)
    print('\nNum of valid set : ', num_valid)

    Arange = list(range(num_total))
    random.shuffle(Arange)  # shuffling the list.

    # copy training images and ground truth into the corresponding folders.
    for i in range(num_train):
        idx = Arange.pop()
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.train_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.train_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_train, prefix='Producing train set:', suffix='Complete', length=50)

    # copy validation images and ground truth into the corresponding folders.
    for i in range(num_valid):
        idx = Arange.pop()
        src = os.path.join(config.origin_data_path, data_list[idx])
        dst = os.path.join(config.valid_path, data_list[idx])
        copyfile(src, dst)

        src = os.path.join(config.origin_GT_path, GT_list[idx])
        dst = os.path.join(config.valid_GT_path, GT_list[idx])
        copyfile(src, dst)

        printProgressBar(i + 1, num_valid, prefix='Producing valid set:', suffix='Complete', length=50)


# take inputs from the user.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ratio of the training and validation set
    parser.add_argument('--valid_ratio', type=float, default=0.3)

    # database path
    parser.add_argument('--origin_data_path', type=str, default='../crack_dataset/training_data/images')
    parser.add_argument('--origin_GT_path', type=str, default='../crack_dataset/training_data/ground_truth')

    # path to training and validation sets.
    parser.add_argument('--train_path', type=str, default='../dataset/train/')
    parser.add_argument('--train_GT_path', type=str, default='../dataset/train_GT/')
    parser.add_argument('--valid_path', type=str, default='../dataset/valid/')
    parser.add_argument('--valid_GT_path', type=str, default='../dataset/valid_GT/')
    config = parser.parse_args()
    print(config)
    main(config)
