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
import traceback
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
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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
# sys.path.append(os.getcwd())
from src.cnn_models import get_model
from src.utils import load_yaml_file, argument_parser, create_logfile, original_labels_to_dataset_labels
# from src.cnn_genomic_datasets import generate_datasets


from src.parse_data import data_split, frequency_calc, merge_haplos_distance
from src.dist_from_freq import calc_dist_each_col
def parse_ms(app_config):

    # load ms files

    # task 1 and 3

    # task 2


    return


def main(app_config):
    try:
        # TODO
        # convert ms to image 1, image 2, image 3 for all window size
        parse_ms(app_config)

        # create dataloader
        train_dataset, val_dataset, test_dataset = generate_datasets(app_config, dataset_type='transfer_learning')

        cnn = CNN_Train_IncResNetv2_Stack(app_config, train_dataset, val_dataset, test_dataset)

        if "only_test" in app_config and app_config['only_test']:

            original_classes, predicted_classes, total_loss, accuracy, prediction_scores = cnn.test()
            cnn.test_report(original_classes, predicted_classes, prediction_scores)
        
        else:
            cnn.train()

            if app_config['test']:

                original_classes, predicted_classes, total_loss, accuracy, prediction_scores = cnn.test()
                cnn.test_report(original_classes, predicted_classes, prediction_scores)

        logging.info('--------------------------------------------------------------------------------')

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)
        logging.error(repr(traceback.extract_tb(e.__traceback__)))


if __name__ == "__main__":
    # Skip the script filename
    # read ms filenames
    data = sys.argv[1:]
    # config_file = 'configs/train_sm3.yaml'
    # app_config = load_yaml_file(config_file)
    # app_config = {
    #     'output_save_path': 'temp_demo_results',
    #     'ms_files': data,
    # }

    app_config = {
        'seed': 42,
        'cnn_model': 'Net_23',
        'optimizer': {'name': 'adam', 'parameters': {'lr': 0.001}},
        
        'output_save_path': 'temp_demo_results',
        'ms_files': data,

        'epochs': 30,
        'batch_size': 20,
        'resume_training': True,
        'load_pretrained_model': f'models/shades_m3/model/model.pt',

        'dataset_name': 'CEU1e8Trunc',
        'w1_window_size': 21,
        'task_1_data_path': 'temp_demo_results/generated_data/w21/freq/',
        'task_2_data_path': 'temp_demo_results/generated_data/w21/freq_dist/',
        'task_3_data_path': 'temp_demo_results/generated_data/w21/w_haplo_d_b/',
        'w2_window_size': 11,
        'w2_task_1_data_path': 'temp_demo_results/generated_data/w11/freq/',
        'w2_task_2_data_path': 'temp_demo_results/generated_data/w11/freq_dist/',
        'w2_task_3_data_path': 'temp_demo_results/generated_data/w11/w_haplo_d_b/',
        'w3_window_size': 51,
        'w3_task_1_data_path': 'temp_demo_results/generated_data/w51/freq/',
        'w3_task_2_data_path': 'temp_demo_results/generated_data/w51/freq_dist/',
        'w3_task_3_data_path': 'temp_demo_results/generated_data/w51/w_haplo_d_b/',
        
        'train_test_split': 4000,
        'train_val_split': 4000,
        'normalization': 'dataset_wise_standardization',
        'early_stop_threshold': 0.1,
        'test': True,
        'validation': False,
        
        'train_dl_num_workers': 7,
        'val_dl_num_workers': 2,
        'test_dl_num_workers': 7, 

        'resize_shape': 299, 
    }

    # create logs
    create_logfile(app_config['output_save_path'])


    logging.info('Starting application')
    # logging.info('Reading config parameters from file {}'.format(config_file))
    logging.info('Application configuration parameters: {}'.format(app_config))

    # https://discuss.pytorch.org/t/reproducibility-dataloader-shuffle-true-using-seeds/173836
    torch.manual_seed(app_config['seed'])

    main(app_config)

    logging.info('--------------------------------------------------------------------------------')

