
## NOTICE ##
# This script is borrowed from the link below with slight changes:
# https://github.com/ternaus/TernausNet

from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision





class TernausNet16(nn.Module):
    def __init__(self, num_classes=1, pretrained=True):
        """
        :param num_classes:
        :param pretrained:
            False - no pre-trained network used
            True - encoder pre-trained with VGG16
        """
        super().__init__()
        self.num_classes = num_classes


        self.encoder = torchvision.models.vgg16(pretrained=pretrained).features


        self.conv1 = nn.Sequential(self.encoder[0],
                                   nn.ReLU(inplace=True),
                                   self.encoder[2],
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(self.encoder[5],
                                   nn.ReLU(inplace=True),
                                   self.encoder[7],
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(self.encoder[10],
                                   nn.ReLU(inplace=True),
                                   self.encoder[12],
                                   nn.ReLU(inplace=True),
                                   self.encoder[14],
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(self.encoder[17],
                                   nn.ReLU(inplace=True),
                                   self.encoder[19],
                                   nn.ReLU(inplace=True),
                                   self.encoder[21],
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(self.encoder[24],
                                   nn.ReLU(inplace=True),
                                   self.encoder[26],
                                   nn.ReLU(inplace=True),
                                   self.encoder[28],
                                   nn.ReLU(inplace=True))

        """
              Paramaters for transposeconv were chosen to avoid artifacts, following
              link https://distill.pub/2016/deconv-checkerboard/
        """
        self.center = nn.Sequential(nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.dec5 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.dec4 = nn.Sequential(nn.Conv2d(768, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(512, 256, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.dec3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(256, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.dec2 = nn.Sequential(nn.Conv2d(192, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True),
                                  nn.ConvTranspose2d(128, 32, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
                                  nn.ReLU(inplace=True))

        self.dec1 = nn.Sequential(nn.Conv2d(96, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                  nn.ReLU(inplace=True))
        self.final = nn.Conv2d(32, num_classes, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        conv1 = self.conv1(x) # 256 - 256
        conv2 = self.conv2(nn.MaxPool2d(2, 2)(conv1)) # 128 - 128
        conv3 = self.conv3(nn.MaxPool2d(2, 2)(conv2)) # 64 - 64
        conv4 = self.conv4(nn.MaxPool2d(2, 2)(conv3)) # 32 - 32
        conv5 = self.conv5(nn.MaxPool2d(2, 2)(conv4)) # 16 - 16

        center = self.center(nn.MaxPool2d(2, 2)(conv5)) # 8 - 16

        dec5 = self.dec5(torch.cat([center, conv5], 1)) # 16 , 16 - 32

        dec4 = self.dec4(torch.cat([dec5, conv4], 1)) # 32,32 - 64
        dec3 = self.dec3(torch.cat([dec4, conv3], 1)) # 64, 64 - 128
        dec2 = self.dec2(torch.cat([dec3, conv2], 1)) # 128, 128 - 256
        dec1 = self.dec1(torch.cat([dec2, conv1], 1)) # 256

        x_out = self.final(dec1) # 256

        return x_out
