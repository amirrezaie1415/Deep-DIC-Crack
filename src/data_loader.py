"""
## NOTICE ##
THIS CODE IS TAKEN FROM THE GITHUB REPOSITORY: https://github.com/LeeJunHyun/Image_Segmentation
HOWEVER, A NUMBER OF CHANGES HAS BEEN MADE AND SEVERAL COMMENTS ARE ADDED.

The functions and classes defined in this script are used to load the images and ground truth files (as batches)
and apply randomly some transformations as a means of data augmentation.
"""

# import necessary modules
import os
import random
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=256, mode='train', augmentation_prob=0.5, pretrained=False):
        """Initializes some attributes"""
        self.root = root
        self.pretrained = pretrained
        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Read, pre-process and return batch of images and ground truth files."""
        image_path = self.image_paths[index]
        filename = image_path.split('/')[-1][:-len(".png")]
        if self.mode in ['train', 'valid']:
            gt_path = self.GT_paths + filename + '_mask.png'
            gt = Image.open(gt_path)
        image = Image.open(image_path).convert('RGB')

        transform = []  # initialize the list of transformations
        if self.mode == 'train':
            if random.random() < self.augmentation_prob:
                angles = [90, 180, 270]
                angle = random.choice(angles)
                image = F.rotate(image, angle)
                gt = F.rotate(gt, angle)
            # Apply horizontal flip and vertical filp.
            if random.random() < self.augmentation_prob:
                image = F.hflip(image)
                gt = F.hflip(gt)
            if random.random() < self.augmentation_prob:
                image = F.vflip(image)
                gt = F.vflip(gt)
            if random.random() < self.augmentation_prob:
                # change the brightness and contrast of the image.
                transform = T.ColorJitter(brightness=0.2, contrast=0.2)
                image = transform(image)
                transform = []

        # Resize image to self.image_size x self.image_size pixels.
        transform.append(T.Resize((self.image_size, self.image_size)))
        # Change image to tensors as the input of the models must be tensors of
        # the size (batch size, channels, height, width)
        transform.append(T.ToTensor())
        # Apply the transformation.
        transform = T.Compose(transform)
        image = transform(image)
        if self.mode in ['train', 'valid']:
            gt = transform(gt)
        # Normalize the images (pixel values range between [-1, +1])
        if self.pretrained:
            norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            norm_ = T.Normalize([0.5], [0.5])
        image = norm_(image)
        if self.mode in ['train', 'valid']:
            return image, gt
        elif self.mode == 'test':
            return image

    def __len__(self):
        """Returns the total number of files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=0, mode='train', augmentation_prob=0.5,
               shuffle_flag=True, pretrained=False):
    """Build and return Dataloader.
    INPUT:
    - image_path = the path to the image
    - image_size = size of the image (as the input of the model, the width and height are the same = 256)
    - batch_size = size of batch of data.
    - num_workers = the number of images that are preprocessed at the same time.
    (for more information about this variable please see
    https://discuss.pytorch.org/t/relation-between-num-workers-batch-size-and-epoch-in-dataloader/18201)
    - mode = it can be "train", "valid" or "test".
    - augmentation_prob = a probability limit based on which some transformations to the data are applied.

    OUTPUT:
    a data loader which is iterable.
    """
    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob,
                          pretrained=pretrained)
    # the shuffle argument in the function below must set to "True" when training, as a result the order of the
    # batches in each epopch will be random.
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle_flag, num_workers=num_workers)
    return data_loader
