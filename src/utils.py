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

sys.path.append('.')
from src.cnn_models import get_model
from src.constants import CEU6k_2_55k_mapping, dataset_metadata
######################################### import end #########################################


def load_yaml_file(file_location):
    """ 
    load a yaml file in python
    """
    with open(file_location, "r") as stream:
        try:
            yaml_data = yaml.safe_load(stream)
            return yaml_data
        except yaml.YAMLError as exc:
            logging.error(exc)
            sys.exit(-1)


def argument_parser():
    """
    parse command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", help="config file path", required=True)

    args = parser.parse_args()

    return args


def create_logfile(save_dir='.', timezone_str='US/Eastern'):
    """
    configure logger
    """

    log_folder = os.path.join(save_dir, 'logs')
    os.makedirs(log_folder, exist_ok = True)

    tz = timezone(timezone_str)
    curr_time = datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')
    filename = 'log_' + str(curr_time) + '.log'
    log_path = os.path.join(log_folder, filename)
    

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(fileHandler)
    # ----- logger configuration


def create_logfile_v2(save_dir='.', timezone_str='US/Eastern', handler_name='app_filehandler'):
    """
    configure logger
    """

    log_folder = os.path.join(save_dir, 'logs')
    os.makedirs(log_folder, exist_ok = True)

    tz = timezone(timezone_str)
    curr_time = datetime.now(tz).strftime('%Y-%m-%d_%H-%M-%S')
    filename = 'log_' + str(curr_time) + '.log'
    log_path = os.path.join(log_folder, filename)
    

    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.set_name(handler_name)
    fileHandler.setFormatter(logFormatter)
    logging.getLogger().addHandler(fileHandler)
    return fileHandler
    # ----- logger configuration
    

def original_labels_to_dataset_labels(labels, dataset_name='CEU5500'):
    """
    labels: list of label
    """

    l_ds = []

    if 'label_mapper' in dataset_metadata[dataset_name]:
        l_mapper_func = dataset_metadata[dataset_name]['label_mapper']
        for l in labels:
            l_ds.append(l_mapper_func[l])
    else:
        l_ds = labels


    return l_ds


def get_color(idx):
    idx = idx * 3
    r = (37 * idx) % 255
    g = (17 * idx) % 255
    b = (29 * idx) % 255
    color = (r / 255.0, g / 255.0, b / 255.0)

    return color


def plot_chromosome_zone(df, chromosome, app_config, zone_name, zone_data, tp_classes, colors_4, report_path):
    if chromosome in app_config['chromosomes']:
        
        ch_df = df[df['chromosome'] == chromosome]

        zoom_start = zone_data['zoom_start']
        zoom_end = zone_data['zoom_end']
        zoomed_df = ch_df[(ch_df['position'] > zoom_start) & (ch_df['position'] < zoom_end)]

        for tp_class, color in zip(tp_classes, colors_4):
            pred_proba = zoomed_df[tp_class]
            chrom_position = zoomed_df['position']

            plt.figure(figsize=(10,6))
            plt.plot(chrom_position, pred_proba, color=color, label=tp_class)
            
            if len(zone_data['axvspan']):
                for i, gene in enumerate(zone_data['axvspan']):

                    gene_start = None
                    gene_end = None
                    gene_name = None
                    color = get_color(i)

                    for key, val in gene.items():
                        if 'start' in key:
                            gene_start = val
                            gene_name = key.split('_')[0]
                        else:
                            gene_end = val
                    
                    # print(gene_name, gene_start, gene_end, color)

                    plt.axvspan(gene_start, gene_end, alpha=0.5, color=color, label=gene_name)
                    # plt.axvspan(mcm6_start, mcm6_end, alpha=0.5, color='#81e5f4', label='MCM6')
            
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.title('Prediction Probability')
            plt.xlabel('Chromosome Position')
            plt.ylabel(f'Prediction Probability of {tp_class} class')
            plt.legend()
            plt.grid()
            # plt.plot()

            file_name = f'{zone_name}_prediction_probability_{tp_class}.pdf'
            plot_full_path = os.path.join(report_path, file_name)
            plt.savefig(plot_full_path)
            plt.close()
            logging.info('{} Prediction Probability of {} plot saved here: {}'.format(zone_name, tp_class, plot_full_path))

