"""
Author: Mohammad Khan
mkhan2024@fau.edu
"""
######################################### import start #########################################
import os
# https://stackoverflow.com/a/74656206
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '24'

import sys
import time
import yaml
import random
from random import randint
import time
import yaml
from argparse import ArgumentParser
import logging
from datetime import datetime
from pytz import timezone
import json
import copy

# # ----- logger configuration
logging.getLogger().setLevel(logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

import numpy as np
import pandas as pd
import cv2


from sklearn.model_selection import train_test_split


import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import models
import pretrainedmodels

######################################### import end #########################################


####################################### model architecture definition start #######################################

class Net_1(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_1')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        # self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, self.output_classes)

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b1(y), 1)
        z = torch.flatten(self.b1(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        # xx = F.relu(self.fc1(xx))
        # xx = F.relu(self.fc2(xx))
        # xx = self.fc3(xx)
        
        xx = self.fc1(xx)

        return xx
    

    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_2(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_2')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        # self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, self.output_classes)

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        # xx = F.relu(self.fc1(xx))
        # xx = F.relu(self.fc2(xx))
        # xx = self.fc3(xx)
        
        xx = self.fc1(xx)

        return xx

    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_3(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net3')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        xx = F.relu(self.fc1(xx))        
        xx = self.fc2(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_4(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_4')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(16, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(16, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),   
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 3, padding=1), 
            nn.ReLU(), 
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(16, 16, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),  
            nn.Conv2d(32, 32, 3, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2),  
        )

        # self.fc1 = nn.Linear(3 * 32 * 4 * 4, 120)
        # self.fc1 = nn.Linear(32 * 4 * 4, 120)
        self.fc1 = nn.Linear(3 * 32 * 8 * 8, 120)
        self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        xx = F.relu(self.fc1(xx))        
        xx = self.fc2(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 64, 64)
        t2_images = torch.randn(1, 1, 64, 64)
        t3_images = torch.randn(1, 1, 64, 64)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_5(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_5')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),    
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 8, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(8, 16, 5),
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        xx = F.relu(self.fc1(xx))        
        xx = self.fc2(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 64, 64)
        t2_images = torch.randn(1, 1, 64, 64)
        t3_images = torch.randn(1, 1, 64, 64)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_6(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_6')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b4 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b5 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b6 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b7 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b8 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b9 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.gap = nn.AvgPool2d(kernel_size = 5, padding = 0)
        # self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, self.output_classes)
        self.fc = nn.Sequential(
            nn.Linear(9 * 16, 120), 
            nn.ReLU(),
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        w11 = torch.flatten(self.gap(self.b1(w11)), 1)
        w12 = torch.flatten(self.gap(self.b2(w12)), 1)
        w13 = torch.flatten(self.gap(self.b3(w13)), 1)

        w21 = torch.flatten(self.gap(self.b4(w21)), 1)
        w22 = torch.flatten(self.gap(self.b5(w22)), 1)
        w23 = torch.flatten(self.gap(self.b6(w23)), 1)

        w31 = torch.flatten(self.gap(self.b7(w31)), 1)
        w32 = torch.flatten(self.gap(self.b8(w32)), 1)
        w33 = torch.flatten(self.gap(self.b9(w33)), 1)

        xx = torch.cat((w11, w12, w13, w21, w22, w23, w31, w32, w33), dim=1)
        
        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                ], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_7(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_7')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        xx = F.relu(self.fc1(xx))        
        xx = self.fc2(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 3, 32, 32)
        t2_images = torch.randn(1, 3, 32, 32)
        t3_images = torch.randn(1, 3, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_8(nn.Module):
    def __init__(self):
        logging.info('Initializing model M1')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 2 * 2, 40)
        self.fc2 = nn.Linear(40, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        xx = F.relu(self.fc1(xx))
        xx = self.fc2(xx)
        
        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_9(nn.Module):
    def __init__(self):
        logging.info('Initializing model M1')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5, stride=2, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5, stride=1, padding=1), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 2 * 2, 40)
        self.fc2 = nn.Linear(40, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        xx = self.fc1(xx)
        xy = F.relu(xx)
        xy = self.fc2(xy)
        
        return xx, xy


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_10(nn.Module):
    def __init__(self):
        logging.info('Initializing model SHADES[M1]')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.m1 = Net_9()
        self.m2 = Net_9()
        self.m3 = Net_9()

        self.fc = nn.Sequential(
            nn.Linear(3 * 40, 120), 
            nn.ReLU(),
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x, _ = self.m1(w11, w12, w13)
        y, _ = self.m2(w21, w22, w23)
        z, _ = self.m3(w31, w32, w33)

        xx = torch.cat((x, y, z), dim=1)

        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_11(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net3')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        # xx = F.relu(self.fc1(xx))        
        # xx = self.fc2(xx)

        xx = self.fc1(xx)
        xy = F.relu(xx)
        xy = self.fc2(xy)

        return xx, xy


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_12(nn.Module):
    def __init__(self):
        logging.info('Initializing model SHADES[Net3]')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.m1 = Net_11()
        self.m2 = Net_11()
        self.m3 = Net_11()

        self.fc = nn.Sequential(
            nn.Linear(3 * 120, 120), 
            nn.ReLU(),
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x, _ = self.m1(w11, w12, w13)
        y, _ = self.m2(w21, w22, w23)
        z, _ = self.m3(w31, w32, w33)

        xx = torch.cat((x, y, z), dim=1)

        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_13(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_13')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = models.inception_v3(weights='IMAGENET1K_V1')
        self.model_in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = self.model
        self.m2 = self.model
        self.m3 = self.model

        self.freeze_models()
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 3, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )


    def forward(self, x, y, z):
        x = torch.flatten(self.m1(x), 1)
        y = torch.flatten(self.m2(y), 1)
        z = torch.flatten(self.m3(z), 1)
        
        xx = torch.cat((x, y, z), dim=1)
        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
        # print(self)
        # print(sm)
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_14(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_14')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = models.inception_v3(weights='IMAGENET1K_V1')
        self.model_in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = self.model
        self.m2 = self.model
        self.m3 = self.model

        self.freeze_models()
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)

        # self.fc = nn.Sequential(
        #     nn.Linear(self.model_in_features * 3, 120), 
        #     nn.ReLU(), 
        #     nn.Linear(120, self.output_classes)
        # )

        self.fc = nn.Linear(self.model_in_features * 3, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.m1(x), 1)
        y = torch.flatten(self.m2(y), 1)
        z = torch.flatten(self.m3(z), 1)
        
        xx = torch.cat((x, y, z), dim=1)
        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
        # print(self)
        # print(sm)
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_15(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_15')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = models.inception_v3(weights='IMAGENET1K_V1')
        self.model_in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = self.model

        self.freeze_models()
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)

        # self.fc = nn.Sequential(
        #     nn.Linear(self.model_in_features * 9, 120), 
        #     nn.ReLU(), 
        #     nn.Linear(120, self.output_classes)
        # )

        self.fc = nn.Linear(self.model_in_features * 9, self.output_classes)


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x_w11 = torch.flatten(self.m1(w11), 1)
        x_w12 = torch.flatten(self.m1(w12), 1)
        x_w13 = torch.flatten(self.m1(w13), 1)

        x_w21 = torch.flatten(self.m1(w21), 1)
        x_w22 = torch.flatten(self.m1(w22), 1)
        x_w23 = torch.flatten(self.m1(w23), 1)

        x_w31 = torch.flatten(self.m1(w31), 1)
        x_w32 = torch.flatten(self.m1(w32), 1)
        x_w33 = torch.flatten(self.m1(w33), 1)
        
        x = torch.cat((x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33), dim=1)
        x = self.fc(x)

        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
        # print(self)
        # print(sm)
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        # for param in self.m2.parameters():
        #     param.requires_grad = False
        
        # for param in self.m3.parameters():
        #     param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_16(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_16')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = models.inception_v3(weights='IMAGENET1K_V1')
        self.model_in_features = self.model.fc.in_features
        self.model.fc = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = self.model

        self.freeze_models()
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)
        # self.freeze_models(self.model_1)

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 9, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )

        # self.fc = nn.Linear(self.model_in_features * 9, self.output_classes)


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x_w11 = torch.flatten(self.m1(w11), 1)
        x_w12 = torch.flatten(self.m1(w12), 1)
        x_w13 = torch.flatten(self.m1(w13), 1)

        x_w21 = torch.flatten(self.m1(w21), 1)
        x_w22 = torch.flatten(self.m1(w22), 1)
        x_w23 = torch.flatten(self.m1(w23), 1)

        x_w31 = torch.flatten(self.m1(w31), 1)
        x_w32 = torch.flatten(self.m1(w32), 1)
        x_w33 = torch.flatten(self.m1(w33), 1)
        
        x = torch.cat((x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33), dim=1)
        x = self.fc(x)

        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
        # print(self)
        # print(sm)
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        # for param in self.m2.parameters():
        #     param.requires_grad = False
        
        # for param in self.m3.parameters():
        #     param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_17(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_17')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        self.model_in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = self.model

        self.freeze_models()

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 9, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x_w11 = torch.flatten(self.m1(w11), 1)
        x_w12 = torch.flatten(self.m1(w12), 1)
        x_w13 = torch.flatten(self.m1(w13), 1)

        x_w21 = torch.flatten(self.m1(w21), 1)
        x_w22 = torch.flatten(self.m1(w22), 1)
        x_w23 = torch.flatten(self.m1(w23), 1)

        x_w31 = torch.flatten(self.m1(w31), 1)
        x_w32 = torch.flatten(self.m1(w32), 1)
        x_w33 = torch.flatten(self.m1(w33), 1)
        
        x = torch.cat((x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33), dim=1)
        x = self.fc(x)

        # return x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33
        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images, t1_images, t2_images, t3_images, t1_images, t2_images, t3_images], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))
        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        # for param in self.m2.parameters():
        #     param.requires_grad = False
        
        # for param in self.m3.parameters():
        #     param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_18(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model Net_18')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        self.model_in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()

        self.freeze_model(self.model)
        
        self.m1 = copy.deepcopy(self.model)
        self.m2 = copy.deepcopy(self.model)
        self.m3 = copy.deepcopy(self.model)
        self.m4 = copy.deepcopy(self.model)
        self.m5 = copy.deepcopy(self.model)
        self.m6 = copy.deepcopy(self.model)
        self.m7 = copy.deepcopy(self.model)
        self.m8 = copy.deepcopy(self.model)
        self.m9 = copy.deepcopy(self.model)

        # self.freeze_models()

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 9, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x_w11 = torch.flatten(self.m1(w11), 1)
        x_w12 = torch.flatten(self.m2(w12), 1)
        x_w13 = torch.flatten(self.m3(w13), 1)

        x_w21 = torch.flatten(self.m4(w21), 1)
        x_w22 = torch.flatten(self.m5(w22), 1)
        x_w23 = torch.flatten(self.m6(w23), 1)

        x_w31 = torch.flatten(self.m7(w31), 1)
        x_w32 = torch.flatten(self.m8(w32), 1)
        x_w33 = torch.flatten(self.m9(w33), 1)
        
        x = torch.cat((x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33), dim=1)
        x = self.fc(x)

        # return x_w11, x_w12, x_w13, x_w21, x_w22, x_w23, x_w31, x_w32, x_w33
        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images, t1_images, t2_images, t3_images, t1_images, t2_images, t3_images], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))
        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        # for param in model.parameters():
        #     param.requires_grad = False
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        # for param in self.m2.parameters():
        #     param.requires_grad = False
        
        # for param in self.m3.parameters():
        #     param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_19(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_3_s_v2')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.b1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b2 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.b3 = nn.Sequential(
            nn.Conv2d(self.input_channels, 6, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(6, 16, 5), 
            nn.ReLU(), 
            nn.MaxPool2d(2, 2), 
        )

        self.fc1 = nn.Linear(3 * 16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, self.output_classes)


    def forward(self, x, y, z):
        x = torch.flatten(self.b1(x), 1)
        y = torch.flatten(self.b2(y), 1)
        z = torch.flatten(self.b3(z), 1)

        xx = torch.cat((x, y, z), dim=1)
        
        # xx = F.relu(self.fc1(xx))        
        # xx = self.fc2(xx)

        xx = self.fc1(xx)
        xy = F.relu(xx)
        # xy = self.fc2(xy)

        return xy


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data=[t1_images, t2_images, t3_images], 
            col_names= [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ]
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))


class Net_20(nn.Module):
    def __init__(self):
        logging.info('Initializing model Net_20')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 1

        self.m1 = Net_19()
        self.m2 = Net_19()
        self.m3 = Net_19()

        self.fc = nn.Sequential(
            nn.Linear(3 * 120, 120), 
            nn.ReLU(),
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x = self.m1(w11, w12, w13)
        y = self.m2(w21, w22, w23)
        z = self.m3(w31, w32, w33)

        xx = torch.cat((x, y, z), dim=1)

        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(1, 1, 32, 32)
        t2_images = torch.randn(1, 1, 32, 32)
        t3_images = torch.randn(1, 1, 32, 32)

        sm = summary(
            self, 
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_21(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model M2')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        self.model_in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()

        # self.freeze_model(self.model)
        
        self.m1 = copy.deepcopy(self.model)
        self.m2 = copy.deepcopy(self.model)
        self.m3 = copy.deepcopy(self.model)

        # self.freeze_models()

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 3, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w1, w2, w3):
        x_w11 = torch.flatten(self.m1(w1), 1)
        x_w12 = torch.flatten(self.m2(w2), 1)
        x_w13 = torch.flatten(self.m3(w3), 1)

        x = torch.cat((x_w11, x_w12, x_w13), dim=1)
        x = self.fc(x)

        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images], 
            device="cpu"
        )

        # logging.info('Model architecture: \n{}'.format(self))
        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        # for param in model.parameters():
        #     param.requires_grad = False
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        # for param in self.m2.parameters():
        #     param.requires_grad = False
        
        # for param in self.m3.parameters():
        #     param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_22(nn.Module):
    def __init__(self):
        logging.info('Initializing transfer learning model M2')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.model = pretrainedmodels.__dict__['inceptionresnetv2'](num_classes=1000, pretrained='imagenet')
        self.model_in_features = self.model.last_linear.in_features
        self.model.last_linear = nn.Identity()

        # self.freeze_model(self.model)
        
        self.m1 = copy.deepcopy(self.model)
        self.m2 = copy.deepcopy(self.model)
        self.m3 = copy.deepcopy(self.model)

        # self.freeze_models()

        self.fc = nn.Sequential(
            nn.Linear(self.model_in_features * 3, 120), 
            nn.ReLU(), 
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w1, w2, w3):
        x_w11 = torch.flatten(self.m1(w1), 1)
        x_w12 = torch.flatten(self.m2(w2), 1)
        x_w13 = torch.flatten(self.m3(w3), 1)

        x = torch.cat((x_w11, x_w12, x_w13), dim=1)
        x = self.fc[1](self.fc[0](x))

        return x


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            input_data = [t1_images, t2_images, t3_images], 
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))
        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_model(self, model):
        model.aux_logits = False
        for param in model.parameters():
            param.requires_grad = False
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_23(nn.Module):
    def __init__(self):
        logging.info('Initializing model SHADES[M2]')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.m1 = Net_22()
        self.m2 = Net_22()
        self.m3 = Net_22()

        self.fc = nn.Sequential(
            nn.Linear(3 * 120, 120), 
            nn.ReLU(),
            nn.Linear(120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x = self.m1(w11, w12, w13)
        y = self.m2(w21, w22, w23)
        z = self.m3(w31, w32, w33)

        xx = torch.cat((x, y, z), dim=1)

        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            device="cpu"
        )

        # logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


class Net_24(nn.Module):
    def __init__(self):
        # s[inception resnet v2] linear model
        logging.info('Initializing model Net_24')
        super().__init__()
        self.output_classes = 4
        self.input_channels = 3

        self.m1 = Net_22()
        self.m2 = Net_22()
        self.m3 = Net_22()

        self.fc = nn.Sequential(
            nn.Linear(3 * 120, self.output_classes)
        )


    def forward(self, w11, w12, w13, w21, w22, w23, w31, w32, w33):
        x = self.m1(w11, w12, w13)
        y = self.m2(w21, w22, w23)
        z = self.m3(w31, w32, w33)

        xx = torch.cat((x, y, z), dim=1)

        xx = self.fc(xx)

        return xx


    def show_summary(self):
        """ show model summary
        https://github.com/TylerYep/torchinfo
        """
        from torchinfo import summary

        t1_images = torch.randn(10, 3, 299, 299)
        t2_images = torch.randn(10, 3, 299, 299)
        t3_images = torch.randn(10, 3, 299, 299)

        sm = summary(
            self, 
            input_data = [
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
                t1_images, t2_images, t3_images, 
            ], 
            col_names = [
                "input_size",
                "kernel_size",
                "output_size",
                "num_params",
            ],
            device="cpu"
        )

        logging.info('Model architecture: \n{}'.format(self))

        logging.info('Model summary: \n{}'.format(sm))
    
    
    def freeze_models(self):
        for param in self.m1.parameters():
            param.requires_grad = False
        
        for param in self.m2.parameters():
            param.requires_grad = False
        
        for param in self.m3.parameters():
            param.requires_grad = False


    def check_not_frozen_layers(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)


####################################### model architecture definition end #######################################


def get_model(app_config):
    model = None
    if app_config['cnn_model'] == 'Net_1':
        model = Net_1()
    elif app_config['cnn_model'] == 'Net_2':
        model = Net_2()
    elif app_config['cnn_model'] == 'Net_3':
        model = Net_3()
    elif app_config['cnn_model'] == 'Net_4':
        model = Net_4()
    elif app_config['cnn_model'] == 'Net_5':
        model = Net_5()
    elif app_config['cnn_model'] == 'Net_6':
        model = Net_6()
    elif app_config['cnn_model'] == 'Net_7':
        model = Net_7()
    elif app_config['cnn_model'] == 'Net_8':
        model = Net_8()
    elif app_config['cnn_model'] == 'Net_10':
        model = Net_10()
    elif app_config['cnn_model'] == 'Net_12':
        model = Net_12()
    elif app_config['cnn_model'] == 'Net_13':
        model = Net_13()
    elif app_config['cnn_model'] == 'Net_14':
        model = Net_14()
    elif app_config['cnn_model'] == 'Net_15':
        model = Net_15()
    elif app_config['cnn_model'] == 'Net_16':
        model = Net_16()
    elif app_config['cnn_model'] == 'Net_17':
        model = Net_17()
    elif app_config['cnn_model'] == 'Net_18':
        model = Net_18()
    elif app_config['cnn_model'] == 'Net_20':
        model = Net_20()
    elif app_config['cnn_model'] == 'Net_21':
        model = Net_21()
    elif app_config['cnn_model'] == 'Net_23':
        model = Net_23()
    elif app_config['cnn_model'] == 'Net_24':
        model = Net_24()
    else:
        logging.warning('Using default model Net_1 as cnn_model parameter is not acceptable!')
        model = Net_1()
    
    return model

