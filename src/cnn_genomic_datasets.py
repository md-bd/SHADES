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
from src.utils import argument_parser, load_yaml_file, create_logfile
from src.constants import *

######################################### import end #########################################


######################################### train dataloader start #########################################

def dataset_details(
    task_data_path,
    skip_file_prefix = 'avg_', 
    save_file = False, 
    dataset_details_file = 'dataset_details.csv',
    ):

    datasets = os.listdir(task_data_path)
    
    logging.info('Available datasets: {}'.format(datasets))
    
    dataset_details = {
        'file_number': [], 
        'task_file_path': [],
        'target_label': []
    }

    for dataset in datasets:
        files = os.listdir(os.path.join(task_data_path, dataset))
        logging.info(f'dataset: {dataset} total files: {len(files)}')

        for file in files:
            if skip_file_prefix in  file:
                logging.info('skipped: {}'.format(file))
                continue
            file_path = os.path.join(os.path.join(task_data_path, dataset, file))
            target_label = dataset
            temp = file.split('/')[-1].split('.')[-2].split('_')
            file_number = int(temp[-1])

            dataset_details['task_file_path'].append(file_path)
            dataset_details['target_label'].append(target_label)
            dataset_details['file_number'].append(file_number)
        

    df_dataset_details = pd.DataFrame.from_dict(dataset_details)
    if save_file:
        df_dataset_details.to_csv(dataset_details_file, index=False)
    
    return df_dataset_details


def get_train_val_test_data(
    app_config,
    ):
    seed = app_config['seed']


    # necessary 3 tasks file paths
    task_3_data_path = app_config['task_3_data_path']
    task3_df = dataset_details(task_3_data_path)
    task3_df = task3_df.rename(columns={"task_file_path": "task3_file_path"})

    task_2_data_path = app_config['task_2_data_path']
    task2_df = dataset_details(task_2_data_path)
    task2_df = task2_df.rename(columns={"task_file_path": "task2_file_path"})

    task_1_data_path = app_config['task_1_data_path']
    task1_df = dataset_details(task_1_data_path)
    task1_df = task1_df.rename(columns={"task_file_path": "task1_file_path"})

    all_task_df = task1_df.merge(
        task2_df, 
        on=['file_number', 'target_label'], 
        how='outer', 
        suffixes=('_1', '_2')
    )

    all_task_df = all_task_df.merge(
        task3_df, 
        on=['file_number', 'target_label'], 
        how='outer', 
        suffixes=('', '_3')
    )

    model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
    model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

    model_total_tasks_files = model_parallel_branches * model_per_branch_channels

    def window_task_df(task_path, task_file_path):
        task_data_path = app_config[task_path]
        task_df = dataset_details(task_data_path)
        task_df = task_df.rename(columns={"task_file_path": task_file_path})
        return task_df

    # if we need extra files.... (for more channels or more branches...)
    if model_total_tasks_files == 9:
        # w2
        w2_task3_df = window_task_df('w2_task_3_data_path', 'w2_task3_file_path')
        all_task_df = all_task_df.merge(
            w2_task3_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_3')
        )

        w2_task2_df = window_task_df('w2_task_2_data_path', 'w2_task2_file_path')
        all_task_df = all_task_df.merge(
            w2_task2_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_2')
        )

        w2_task1_df = window_task_df('w2_task_1_data_path', 'w2_task1_file_path')
        all_task_df = all_task_df.merge(
            w2_task1_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_1')
        )

        # w3
        w3_task3_df = window_task_df('w3_task_3_data_path', 'w3_task3_file_path')
        all_task_df = all_task_df.merge(
            w3_task3_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_3')
        )

        w3_task2_df = window_task_df('w3_task_2_data_path', 'w3_task2_file_path')
        all_task_df = all_task_df.merge(
            w3_task2_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_2')
        )

        w3_task1_df = window_task_df('w3_task_1_data_path', 'w3_task1_file_path')
        all_task_df = all_task_df.merge(
            w3_task1_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_1')
        )

    if model_total_tasks_files == 3:
        ttv_x = all_task_df[['task3_file_path', 'task2_file_path', 'task1_file_path']]
        ttv_y = all_task_df['target_label']
    
    elif model_total_tasks_files == 9:
        ttv_x = all_task_df[
            [
            'task3_file_path', 'task2_file_path', 'task1_file_path', 
            'w2_task3_file_path', 'w2_task2_file_path', 'w2_task1_file_path', 
            'w3_task3_file_path', 'w3_task2_file_path', 'w3_task1_file_path'
            ]
        ]
        ttv_y = all_task_df['target_label']
    
    else: 
        logging.error('no code for model_total_tasks_files {}'.format(model_total_tasks_files))
    

    # test, train_val split
    train_test_split_size = app_config['train_test_split']
    X_tv, X_test, y_tv, y_test = train_test_split(
        ttv_x, ttv_y, 
        test_size=train_test_split_size, random_state=seed, 
        stratify=ttv_y
    )

    # train, val split
    train_val_split_size = app_config['train_val_split']
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, 
        test_size=train_val_split_size, random_state=seed, 
        stratify=y_tv
    )


    logging.info('X_tv {}'.format(X_tv.shape))
    logging.info('y_tv {}'.format(y_tv.shape))
    logging.info('X_test   {}'.format(X_test.shape))
    logging.info('y_test   {}'.format(y_test.shape))
    logging.info('X_train  {}'.format(X_train.shape))
    logging.info('y_train  {}'.format(y_train.shape))
    logging.info('X_val    {}'.format(X_val.shape))
    logging.info('y_val    {}'.format(y_val.shape))

    if "only_test" in app_config and app_config['only_test']:
        logging.info('Creating only test dataset. Test data will contain all data.')
        logging.info('ttv_x {}'.format(ttv_x.shape))
        logging.info('ttv_y {}'.format(ttv_y.shape))
        return ttv_x, ttv_y, X_train, y_train, X_val, y_val

    if app_config['validation']:
        return X_test, y_test, X_train, y_train, X_val, y_val
    else:
        return X_test, y_test, X_tv, y_tv, X_val, y_val



def get_train_val_test_tfln_data(
        app_config,
    ):
    
    seed = app_config['seed']


    # necessary 3 tasks file paths
    task_3_data_path = app_config['task_3_data_path']
    task3_df = dataset_details(task_3_data_path)
    task3_df = task3_df.rename(columns={"task_file_path": "task3_file_path"})

    task_2_data_path = app_config['task_2_data_path']
    task2_df = dataset_details(task_2_data_path)
    task2_df = task2_df.rename(columns={"task_file_path": "task2_file_path"})

    task_1_data_path = app_config['task_1_data_path']
    task1_df = dataset_details(task_1_data_path)
    task1_df = task1_df.rename(columns={"task_file_path": "task1_file_path"})

    all_task_df = task1_df.merge(
        task2_df, 
        on=['file_number', 'target_label'], 
        how='outer', 
        suffixes=('_1', '_2')
    )

    all_task_df = all_task_df.merge(
        task3_df, 
        on=['file_number', 'target_label'], 
        how='outer', 
        suffixes=('', '_3')
    )

    model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
    model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

    model_total_tasks_files = model_parallel_branches * model_per_branch_channels

    def window_task_df(task_path, task_file_path):
        task_data_path = app_config[task_path]
        task_df = dataset_details(task_data_path)
        task_df = task_df.rename(columns={"task_file_path": task_file_path})
        return task_df

    # if we need extra files.... (for more channels or more branches...)
    if model_total_tasks_files == 27:
        # w2
        w2_task3_df = window_task_df('w2_task_3_data_path', 'w2_task3_file_path')
        all_task_df = all_task_df.merge(
            w2_task3_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_3')
        )

        w2_task2_df = window_task_df('w2_task_2_data_path', 'w2_task2_file_path')
        all_task_df = all_task_df.merge(
            w2_task2_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_2')
        )

        w2_task1_df = window_task_df('w2_task_1_data_path', 'w2_task1_file_path')
        all_task_df = all_task_df.merge(
            w2_task1_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w2_1')
        )

        # w3
        w3_task3_df = window_task_df('w3_task_3_data_path', 'w3_task3_file_path')
        all_task_df = all_task_df.merge(
            w3_task3_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_3')
        )

        w3_task2_df = window_task_df('w3_task_2_data_path', 'w3_task2_file_path')
        all_task_df = all_task_df.merge(
            w3_task2_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_2')
        )

        w3_task1_df = window_task_df('w3_task_1_data_path', 'w3_task1_file_path')
        all_task_df = all_task_df.merge(
            w3_task1_df, 
            on=['file_number', 'target_label'], 
            how='outer', 
            suffixes=('', 'w3_1')
        )
    
    
    if model_total_tasks_files == 9:
        ttv_x = all_task_df[['task3_file_path', 'task2_file_path', 'task1_file_path']]
        ttv_y = all_task_df['target_label']
    
    elif model_total_tasks_files == 27:
        ttv_x = all_task_df[
            [
            'task3_file_path', 'task2_file_path', 'task1_file_path', 
            'w2_task3_file_path', 'w2_task2_file_path', 'w2_task1_file_path', 
            'w3_task3_file_path', 'w3_task2_file_path', 'w3_task1_file_path'
            ]
        ]
        ttv_y = all_task_df['target_label']
    
    else: 
        logging.error('no code for model_total_tasks_files {}'.format(model_total_tasks_files))
    

    # test, train_val split
    train_test_split_size = app_config['train_test_split']
    X_tv, X_test, y_tv, y_test = train_test_split(
        ttv_x, ttv_y, 
        test_size=train_test_split_size, random_state=seed, 
        stratify=ttv_y
    )

    # train, val split
    train_val_split_size = app_config['train_val_split']
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, 
        test_size=train_val_split_size, random_state=seed, 
        stratify=y_tv
    )


    logging.info('X_tv {}'.format(X_tv.shape))
    logging.info('y_tv {}'.format(y_tv.shape))
    logging.info('X_test   {}'.format(X_test.shape))
    logging.info('y_test   {}'.format(y_test.shape))
    logging.info('X_train  {}'.format(X_train.shape))
    logging.info('y_train  {}'.format(y_train.shape))
    logging.info('X_val    {}'.format(X_val.shape))
    logging.info('y_val    {}'.format(y_val.shape))

    if "only_test" in app_config and app_config['only_test']:
        logging.info('Creating only test dataset. Test data will contain all data.')
        logging.info('ttv_x {}'.format(ttv_x.shape))
        logging.info('ttv_y {}'.format(ttv_y.shape))
        return ttv_x, ttv_y, X_train, y_train, X_val, y_val

    if app_config['validation']:
        return X_test, y_test, X_train, y_train, X_val, y_val
    else:
        return X_test, y_test, X_tv, y_tv, X_val, y_val


class GenomicDataset(Dataset):
    """Genomic Dataset """
    def __init__(self, x_dataframe, y_dataframe, app_config, resize_shape=(32, 32), transform=None):
        """
        """
        self.app_config = app_config
        self.x_df = x_dataframe
        self.y_df = y_dataframe

        self.resize = resize_shape

        self.model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
        self.model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels

        # self.model_parallel_channels = cnn_model_metadata[self.app_config['cnn_model']]['parallel_channels']
        self.transform_function_init()

        self.init_labels()

        # if 'datasets' in app_config and app_config['datasets'] != '':
        #     self.labels = app_config['datasets']
        # else:
        #     self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        # # self.labels = ['Neut', 'BalSel', 'AI', 'Sweeps']
        # # self.le = LabelEncoder()
        # # self.le.fit(self.labels)

        # self.lb = LabelBinarizer()
        # self.lb.fit(self.labels) # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


    def __len__(self):
        return len(self.x_df)
    

    def __getitem__(self, idx):
        # print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
            logging.warning('index to list happened...')
        
        # ['task3_file_path', 'task2_file_path', 'task1_file_path']
        task_1_img_location = self.x_df.iloc[idx]['task1_file_path']
        task_2_img_location = self.x_df.iloc[idx]['task2_file_path']
        task_3_img_location = self.x_df.iloc[idx]['task3_file_path']

        # print(task_1_img_location)

        # label = self.le.transform([self.y_df.iloc[idx]])
        label = self.lb.transform([self.y_df.iloc[idx]])

        # return task_1_img_location, task_2_img_location, task_3_img_location, label

        # load and resize the images
        ## task 1 data
        torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
        ## task 2 data
        torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
        ## task 3 data
        torch_t3_data = self.read_and_preprocess_data(task_3_img_location)
        # return torch_t2_data, torch_t3_data, label


        if self.model_total_tasks_files == 9:
            # w2
            task_1_img_location = self.x_df.iloc[idx]['w2_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w2_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w2_task3_file_path']

            # load and resize the images
            ## task 1 data
            w2_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            ## task 2 data
            w2_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            ## task 3 data
            w2_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            # w3
            task_1_img_location = self.x_df.iloc[idx]['w3_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w3_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w3_task3_file_path']

            # load and resize the images
            ## task 1 data
            w3_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            ## task 2 data
            w3_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            ## task 3 data
            w3_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)
        

        # add channels for necessary configs models
        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            torch_t1_data = torch.cat((torch_t1_data, w2_torch_t1_data, w3_torch_t1_data), 0)
            torch_t2_data = torch.cat((torch_t2_data, w2_torch_t2_data, w3_torch_t2_data), 0)
            torch_t3_data = torch.cat((torch_t3_data, w2_torch_t3_data, w3_torch_t3_data), 0)


        # transform data
        if self.transform:

            if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
                torch_t1_data = self.transform(torch_t1_data, window=1, task=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, window=1, task=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, window=1, task=3, resize_shape=self.resize)

            elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:
                torch_t1_data = self.transform(torch_t1_data, window=1, task=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, window=1, task=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, window=1, task=3, resize_shape=self.resize)

                w2_torch_t1_data = self.transform(w2_torch_t1_data, window=2, task=1, resize_shape=self.resize)
                w2_torch_t2_data = self.transform(w2_torch_t2_data, window=2, task=2, resize_shape=self.resize)
                w2_torch_t3_data = self.transform(w2_torch_t3_data, window=2, task=3, resize_shape=self.resize)

                w3_torch_t1_data = self.transform(w3_torch_t1_data, window=3, task=1, resize_shape=self.resize)
                w3_torch_t2_data = self.transform(w3_torch_t2_data, window=3, task=2, resize_shape=self.resize)
                w3_torch_t3_data = self.transform(w3_torch_t3_data, window=3, task=3, resize_shape=self.resize)
            
            elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                torch_t1_data = self.transform(torch_t1_data, branch=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, branch=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, branch=3, resize_shape=self.resize)
            
            else:
                logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                    self.model_parallel_branches, self.model_per_branch_channels))

        # return data
        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
            return torch_t1_data, torch_t2_data, torch_t3_data, label
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:
            return torch_t1_data, torch_t2_data, torch_t3_data, \
                w2_torch_t1_data, w2_torch_t2_data, w2_torch_t3_data, \
                w3_torch_t1_data, w3_torch_t2_data, w3_torch_t3_data, \
                label
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            return torch_t1_data, torch_t2_data, torch_t3_data, label
        
        else:
            logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                self.model_parallel_branches, self.model_per_branch_channels))
            return -1


    def transform_function_init(self):
        # app config
        cnn_model = self.app_config['cnn_model']
        cnn_model_config = cnn_model_metadata[cnn_model]
        input_size = cnn_model_config['input_size']

        # find dataset metadata for means and stds
        if 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            self.dataset_name = self.app_config['dataset_name']
        else:
            self.dataset_name = 'CEU6000'
        logging.info(f'Using {self.dataset_name} dataset for dataset mean and std.')
        
        ds_metadata = dataset_metadata[self.dataset_name]['dataset_metadata']

        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                w1_means_stds['means'], 
                w1_means_stds['means'], 
                w1_means_stds['means'],
            ]
            self.stds = [
                [-1], 
                w1_means_stds['stds'], 
                w1_means_stds['stds'], 
                w1_means_stds['stds'],
            ]
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:

            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]
            
            window_size = self.app_config['w2_window_size']
            # input_size = 32
            stride = 10
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            # input_size = 32
            stride = 10
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                w1_means_stds['means'], 
                w2_means_stds['means'], 
                w3_means_stds['means'],
            ]
            self.stds = [
                [-1], 
                w1_means_stds['stds'], 
                w2_means_stds['stds'], 
                w3_means_stds['stds'],
            ]
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w2_window_size']
            # input_size = 32
            stride = 10
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            # input_size = 32
            stride = 10
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]


            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w2_means_stds['means'][1], w3_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w2_means_stds['means'][2], w3_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w2_means_stds['means'][3], w3_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w2_means_stds['stds'][1], w3_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w2_means_stds['stds'][2], w3_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w2_means_stds['stds'][3], w3_means_stds['stds'][3]], (-1, 1, 1)), 
            ]

        else:
            logging.error('No code for model_parallel_branches {} and model_per_branch_channels {}'.format(self.self.model_parallel_branches, self.model_per_branch_channels))


        transform_func = None
        if self.app_config['normalization'] != '':
            # normalization: 'image_wise_standardization'
            if self.app_config['normalization'] == 'image_wise_standardization':
                transform_func = self.transform_image_wise_standardization
                logging.info('using transform_image_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'image_wise_normalizaton':
                transform_func = self.transform_image_wise_normalization
                logging.info('using transform_image_wise_normalization function for data normalization process')

            elif self.app_config['normalization'] == 'dataset_wise_standardization':

                # for br3, ch3 => we need to select different one
                if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                    transform_func = self.transform_3channel_3branch_standardization
                else:
                    transform_func = self.transform_dataset_wise_standardization
                logging.info('using transform_dataset_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'histogram_equalization':
                if self.model_per_branch_channels == 1:
                    transform_func = self.transform_histogram_equalization
                else:
                    logging.error(f'no function description for {self.app_config["normalization"]}')
                    logging.info('Using no normalization in data')

            else:
                logging.warning(f'no function description for {self.app_config["normalization"]}')
                logging.info('Using no normalization in data')

        self.transform = transform_func

    
    def transform_3channel_3branch_standardization(self, data, **kwargs):
        """
        """

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'branch' in kwargs:
            logging.error('expected branch value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        # task = int(kwargs['task'])
        branch = int(kwargs['branch'])
        # standardized data
        s_data = (data - self.means[branch]) / (self.stds[branch])

        return s_data.float()


    def transform_dataset_wise_standardization(self, data, **kwargs):
        """
        https://www.geeksforgeeks.org/args-kwargs-python/
        """
        # logging.info('means: {}'.format(means))

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'task' in kwargs:
            logging.error('expected task value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        task = int(kwargs['task'])
        window = int(kwargs['window'])
        # standardized data
        s_data = (data - self.means[window][task]) / (self.stds[window][task])

        return s_data


    def transform_image_wise_standardization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate mean and std
        # https://pytorch.org/docs/stable/generated/torch.std_mean.html
        # dim 2, 3 are 32x32 image dim in batch
        # t_std, t_mean  = torch.std_mean(data, dim=(2,3), keepdim=True)
        # single image case, dim 1, 2 are 32x32 image dim
        t_std, t_mean  = torch.std_mean(data, dim=(1,2), keepdim=True)

        # normalize data
        s_data = (data - t_mean) / t_std

        return s_data


    def transform_image_wise_normalization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate min and max
        t_min, t_max  = data.min(), data.max()

        # normalize data
        s_data = (data - t_min) / (t_max - t_min)

        return s_data


    def transform_histogram_equalization(self, data, **kwargs):
        """
        """
        def normalize8(img):
            mn = img.min()
            mx = img.max()
            mx -= mn
            img = ((img - mn)/ mx) * 255
            return img.astype(np.uint8)

        # convert to numpy 
        data = data.squeeze().cpu().detach().numpy() # (32, 32)
        task = int(kwargs['task'])
        if task == 1:
            data = data.astype(np.uint8)
        else:
            data = normalize8(data)
        
        data = cv2.equalizeHist(data) / 255.0

        # convert to torch
        data = self.convert_numpy_to_torch(data) # (1, 32, 32)
        return data
    

    def read_and_preprocess_data(self, task_img_location, dtype=None):
        
        task_data = pd.read_csv(task_img_location)
        if dtype is not None:
            task_data = task_data.to_numpy(dtype=dtype)
        else:
            task_data = task_data.to_numpy()
        # print(type(data_2), data_2.shape)

        # add condition for none resize
        if self.resize is not None:
            # interpolation ref
            # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
            if 'resize_interpolation' in self.app_config and self.app_config['resize_interpolation'] != '':
                if self.app_config['resize_interpolation'] == 'same':
                    task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                elif self.app_config['resize_interpolation'] == 'shape_dependent':
                    if task_data.shape[0] > self.resize[0]:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                    else:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
            else:
                task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
        
        # torch_task_data = torch.from_numpy(task_data)
        # torch_task_data = torch_task_data.type(dtype=torch.float32)
        # torch_task_data = torch_task_data.unsqueeze(0)
        torch_task_data = self.convert_numpy_to_torch(task_data=task_data)

        return torch_task_data
    

    def convert_numpy_to_torch(self, task_data):
        torch_task_data = torch.from_numpy(task_data)
        torch_task_data = torch_task_data.type(dtype=torch.float32)
        torch_task_data = torch_task_data.unsqueeze(0)

        return torch_task_data


    def init_labels(self):
        if 'datasets' in self.app_config and self.app_config['datasets'] != '':
            self.labels = self.app_config['datasets']
        
        elif 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            if self.app_config['dataset_name'] in dataset_metadata.keys():
                self.labels = dataset_metadata[self.app_config['dataset_name']]['labels']
            else:
                self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        else:
            self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']

        logging.info(f'Using labels: {self.labels}')

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)    # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


class GenomicTLDataset(Dataset):
    """Genomic Transfer Learning Dataset """
    def __init__(self, x_dataframe, y_dataframe, app_config, resize_shape=(299, 299), transform=None):
        """
        """
        self.app_config = app_config
        self.x_df = x_dataframe
        self.y_df = y_dataframe

        self.resize = resize_shape

        self.model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
        self.model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels

        # self.model_parallel_channels = cnn_model_metadata[self.app_config['cnn_model']]['parallel_channels']
        self.transform_function_init()

        self.init_labels()


    def __len__(self):
        return len(self.x_df)
    

    def __getitem__(self, idx):
        # print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()
            logging.warning('index to list happened...')
        
        task_1_img_location = self.x_df.iloc[idx]['task1_file_path']
        task_2_img_location = self.x_df.iloc[idx]['task2_file_path']
        task_3_img_location = self.x_df.iloc[idx]['task3_file_path']

        label = self.lb.transform([self.y_df.iloc[idx]])

        # load and resize the images
        torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
        torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
        torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

        # copy same data from channels for each branch
        torch_t1_data = torch.cat((torch_t1_data, torch_t1_data, torch_t1_data), 0)
        torch_t2_data = torch.cat((torch_t2_data, torch_t2_data, torch_t2_data), 0)
        torch_t3_data = torch.cat((torch_t3_data, torch_t3_data, torch_t3_data), 0)

        if self.model_total_tasks_files == 27:
            # w2
            task_1_img_location = self.x_df.iloc[idx]['w2_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w2_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w2_task3_file_path']

            # load and resize the images
            w2_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            w2_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            w2_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            w2_torch_t1_data = torch.cat((w2_torch_t1_data, w2_torch_t1_data, w2_torch_t1_data), 0)
            w2_torch_t2_data = torch.cat((w2_torch_t2_data, w2_torch_t2_data, w2_torch_t2_data), 0)
            w2_torch_t3_data = torch.cat((w2_torch_t3_data, w2_torch_t3_data, w2_torch_t3_data), 0)

            # w3
            task_1_img_location = self.x_df.iloc[idx]['w3_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w3_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w3_task3_file_path']

            # load and resize the images
            w3_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            w3_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            w3_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            w3_torch_t1_data = torch.cat((w3_torch_t1_data, w3_torch_t1_data, w3_torch_t1_data), 0)
            w3_torch_t2_data = torch.cat((w3_torch_t2_data, w3_torch_t2_data, w3_torch_t2_data), 0)
            w3_torch_t3_data = torch.cat((w3_torch_t3_data, w3_torch_t3_data, w3_torch_t3_data), 0)


        # transform data
        if self.transform:
            torch_t1_data = self.transform(torch_t1_data, branch=1, resize_shape=self.resize)
            torch_t2_data = self.transform(torch_t2_data, branch=2, resize_shape=self.resize)
            torch_t3_data = self.transform(torch_t3_data, branch=3, resize_shape=self.resize)

            if self.model_total_tasks_files == 27:

                w2_torch_t1_data = self.transform(w2_torch_t1_data, branch=4, resize_shape=self.resize)
                w2_torch_t2_data = self.transform(w2_torch_t2_data, branch=5, resize_shape=self.resize)
                w2_torch_t3_data = self.transform(w2_torch_t3_data, branch=6, resize_shape=self.resize)

                w3_torch_t1_data = self.transform(w3_torch_t1_data, branch=7, resize_shape=self.resize)
                w3_torch_t2_data = self.transform(w3_torch_t2_data, branch=8, resize_shape=self.resize)
                w3_torch_t3_data = self.transform(w3_torch_t3_data, branch=9, resize_shape=self.resize)


        # return data
        if self.model_total_tasks_files == 27:
            return torch_t1_data, torch_t2_data, torch_t3_data, \
                w2_torch_t1_data, w2_torch_t2_data, w2_torch_t3_data, \
                w3_torch_t1_data, w3_torch_t2_data, w3_torch_t3_data, \
                label
        else:
            return torch_t1_data, torch_t2_data, torch_t3_data, label


    def transform_function_init(self):
        # app config
        cnn_model = self.app_config['cnn_model']
        cnn_model_config = cnn_model_metadata[cnn_model]
        input_size = cnn_model_config['input_size']

        # find dataset metadata for means and stds
        if 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            self.dataset_name = self.app_config['dataset_name']
        else:
            self.dataset_name = 'CEU6000'
        logging.info(f'Using {self.dataset_name} dataset for dataset mean and std.')
        
        ds_metadata = dataset_metadata[self.dataset_name]['dataset_metadata']

        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]


            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w1_means_stds['means'][1], w1_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w1_means_stds['means'][2], w1_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w1_means_stds['means'][3], w1_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w1_means_stds['stds'][1], w1_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w1_means_stds['stds'][2], w1_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w1_means_stds['stds'][3], w1_means_stds['stds'][3]], (-1, 1, 1)), 
            ]
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 3:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w2_window_size']
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w1_means_stds['means'][1], w1_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w1_means_stds['means'][2], w1_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w1_means_stds['means'][3], w1_means_stds['means'][3]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][1], w2_means_stds['means'][1], w2_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][2], w2_means_stds['means'][2], w2_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][3], w2_means_stds['means'][3], w2_means_stds['means'][3]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][1], w3_means_stds['means'][1], w3_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][2], w3_means_stds['means'][2], w3_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][3], w3_means_stds['means'][3], w3_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w1_means_stds['stds'][1], w1_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w1_means_stds['stds'][2], w1_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w1_means_stds['stds'][3], w1_means_stds['stds'][3]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][1], w2_means_stds['stds'][1], w2_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][2], w2_means_stds['stds'][2], w2_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][3], w2_means_stds['stds'][3], w2_means_stds['stds'][3]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][1], w3_means_stds['stds'][1], w3_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][2], w3_means_stds['stds'][2], w3_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][3], w3_means_stds['stds'][3], w3_means_stds['stds'][3]], (-1, 1, 1)), 
            ]
        
        else:
            logging.error('No code for model_parallel_branches {} and model_per_branch_channels {}'.format(self.self.model_parallel_branches, self.model_per_branch_channels))



        transform_func = None
        if self.app_config['normalization'] != '':
            # normalization: 'image_wise_standardization'
            if self.app_config['normalization'] == 'image_wise_standardization':
                #TODO
                transform_func = self.transform_image_wise_standardization
                logging.info('using transform_image_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'image_wise_normalizaton':
                #TODO
                transform_func = self.transform_image_wise_normalization
                logging.info('using transform_image_wise_normalization function for data normalization process')

            elif self.app_config['normalization'] == 'dataset_wise_standardization':

                # for br3, ch3 => we need to select different one
                # if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                #     transform_func = self.transform_3channel_3branch_standardization
                # else:
                #     transform_func = self.transform_dataset_wise_standardization
                transform_func = self.transform_3channel_3branch_standardization
                logging.info('using transform_dataset_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'histogram_equalization':
                #TODO
                if self.model_per_branch_channels == 1:
                    transform_func = self.transform_histogram_equalization
                else:
                    logging.error(f'no function description for {self.app_config["normalization"]}')
                    logging.info('Using no normalization in data')

            else:
                logging.warning(f'no function description for {self.app_config["normalization"]}')
                logging.info('Using no normalization in data')

        self.transform = transform_func

    
    def transform_3channel_3branch_standardization(self, data, **kwargs):
        """
        """

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'branch' in kwargs:
            logging.error('expected branch value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        # task = int(kwargs['task'])
        branch = int(kwargs['branch'])
        # standardized data
        s_data = (data - self.means[branch]) / (self.stds[branch])

        return s_data.float()


    def transform_dataset_wise_standardization(self, data, **kwargs):
        """
        https://www.geeksforgeeks.org/args-kwargs-python/
        """
        # logging.info('means: {}'.format(means))

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'task' in kwargs:
            logging.error('expected task value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        task = int(kwargs['task'])
        window = int(kwargs['window'])
        # standardized data
        s_data = (data - self.means[window][task]) / (self.stds[window][task])

        return s_data


    def transform_image_wise_standardization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate mean and std
        # https://pytorch.org/docs/stable/generated/torch.std_mean.html
        # dim 2, 3 are 32x32 image dim in batch
        # t_std, t_mean  = torch.std_mean(data, dim=(2,3), keepdim=True)
        # single image case, dim 1, 2 are 32x32 image dim
        t_std, t_mean  = torch.std_mean(data, dim=(1,2), keepdim=True)

        # normalize data
        s_data = (data - t_mean) / t_std

        return s_data


    def transform_image_wise_normalization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate min and max
        t_min, t_max  = data.min(), data.max()

        # normalize data
        s_data = (data - t_min) / (t_max - t_min)

        return s_data


    def transform_histogram_equalization(self, data, **kwargs):
        """
        """
        def normalize8(img):
            mn = img.min()
            mx = img.max()
            mx -= mn
            img = ((img - mn)/ mx) * 255
            return img.astype(np.uint8)

        # convert to numpy 
        data = data.squeeze().cpu().detach().numpy() # (32, 32)
        task = int(kwargs['task'])
        if task == 1:
            data = data.astype(np.uint8)
        else:
            data = normalize8(data)
        
        data = cv2.equalizeHist(data) / 255.0

        # convert to torch
        data = self.convert_numpy_to_torch(data) # (1, 32, 32)
        return data
    

    def read_and_preprocess_data(self, task_img_location, dtype=None):
        
        task_data = pd.read_csv(task_img_location)
        if dtype is not None:
            task_data = task_data.to_numpy(dtype=dtype)
        else:
            task_data = task_data.to_numpy()
        # print(type(data_2), data_2.shape)

        # add condition for none resize
        if self.resize is not None:
            # interpolation ref
            # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
            if 'resize_interpolation' in self.app_config and self.app_config['resize_interpolation'] != '':
                if self.app_config['resize_interpolation'] == 'same':
                    task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                elif self.app_config['resize_interpolation'] == 'shape_dependent':
                    if task_data.shape[0] > self.resize[0]:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                    else:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
            else:
                task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
        
        # torch_task_data = torch.from_numpy(task_data)
        # torch_task_data = torch_task_data.type(dtype=torch.float32)
        # torch_task_data = torch_task_data.unsqueeze(0)
        torch_task_data = self.convert_numpy_to_torch(task_data=task_data)

        return torch_task_data
    

    def convert_numpy_to_torch(self, task_data):
        torch_task_data = torch.from_numpy(task_data)
        torch_task_data = torch_task_data.type(dtype=torch.float32)
        torch_task_data = torch_task_data.unsqueeze(0)

        return torch_task_data


    def init_labels(self):
        if 'datasets' in self.app_config and self.app_config['datasets'] != '':
            self.labels = self.app_config['datasets']
        
        elif 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            if self.app_config['dataset_name'] in dataset_metadata.keys():
                self.labels = dataset_metadata[self.app_config['dataset_name']]['labels']
            else:
                self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        else:
            self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']

        logging.info(f'Using labels: {self.labels}')

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)    # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


def generate_datasets(app_config, dataset_type='genomic'):

    # get data
    if dataset_type == 'genomic':
        X_test, y_test, X_train, y_train, X_val, y_val = get_train_val_test_data(app_config)
    elif dataset_type =='transfer_learning':
        X_test, y_test, X_train, y_train, X_val, y_val = get_train_val_test_tfln_data(app_config)
    else:
        logging.error('Dataset type not found!')


    dataset_ref_path = os.path.join(app_config['output_save_path'], 'dataset')
    os.makedirs(dataset_ref_path, exist_ok = True)

    ### save train, test, val dataset for ref
    # train -------------------
    train_path = os.path.join(dataset_ref_path, 'x_train.csv')
    X_train.to_csv(train_path, index=False)
    train_path = os.path.join(dataset_ref_path, 'y_train.csv')
    y_train.to_csv(train_path, index=False)
    #-------------------

    # test -------------------
    test_path = os.path.join(dataset_ref_path, 'x_test.csv')
    X_test.to_csv(test_path, index=False)
    test_path = os.path.join(dataset_ref_path, 'y_test.csv')
    y_test.to_csv(test_path, index=False)
    #-------------------

    # val -------------------
    val_path = os.path.join(dataset_ref_path, 'x_val.csv')
    X_val.to_csv(val_path, index=False)
    val_path = os.path.join(dataset_ref_path, 'y_val.csv')
    y_val.to_csv(val_path, index=False)
    #-------------------

    resize_shape = (32, 32)
    if 'resize_shape' in app_config:
        temp_r_shape = int(app_config['resize_shape'])
        resize_shape = (temp_r_shape, temp_r_shape)

    elif 'input_size' in cnn_model_metadata[app_config['cnn_model']]:
        temp_r_shape = int(cnn_model_metadata[app_config['cnn_model']]['input_size'])
        resize_shape = (temp_r_shape, temp_r_shape)
    
    else:
        logging.warning('using default resize shape')
    

    logging.info('resize shape used : {}'.format(resize_shape))

    dataset = None
    if dataset_type == 'genomic':
        dataset = GenomicDataset
        logging.info('Using Genomic Dataset as Dataset')
    elif dataset_type == 'transfer_learning':
        dataset = GenomicTLDataset
        logging.info('Using Genomic Transfer Learning Dataset as Dataset')
    else:
        logging.error('Dataset type not found!')


    train_dataset = dataset(
        X_train, y_train, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    val_dataset = dataset(
        X_val, y_val, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    test_dataset = dataset(
        X_test, y_test, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    return train_dataset, val_dataset, test_dataset


def generate_datasets_fm_file(app_config, dataset_ref_path, dataset_type='genomic'):
    """
    load csv dataset path file directly to dataloader. No need to create train, val, test split
    """

    ## load csv files from dataset_ref_path

    # if dataset has index, index_col = 0 should be there in read_csv
    # train -------------------
    train_path = os.path.join(dataset_ref_path, 'x_train.csv')
    X_train = pd.read_csv(train_path)
    train_path = os.path.join(dataset_ref_path, 'y_train.csv')
    y_train = pd.read_csv(train_path)


    # test -------------------
    test_path = os.path.join(dataset_ref_path, 'x_test.csv')
    X_test = pd.read_csv(test_path)
    test_path = os.path.join(dataset_ref_path, 'y_test.csv')
    y_test = pd.read_csv(test_path)


    # val -------------------
    val_path = os.path.join(dataset_ref_path, 'x_val.csv')
    X_val = pd.read_csv(val_path)
    val_path = os.path.join(dataset_ref_path, 'y_val.csv')
    y_val = pd.read_csv(val_path)

    resize_shape = (32, 32)
    if 'resize_shape' in app_config:
        temp_r_shape = int(app_config['resize_shape'])
        resize_shape = (temp_r_shape, temp_r_shape)

    elif 'input_size' in cnn_model_metadata[app_config['cnn_model']]:
        temp_r_shape = int(cnn_model_metadata[app_config['cnn_model']]['input_size'])
        resize_shape = (temp_r_shape, temp_r_shape)
    
    else:
        logging.warning('using default resize shape')
    

    logging.info('resize shape used : {}'.format(resize_shape))

    dataset = None
    if dataset_type == 'genomic':
        dataset = GenomicDataset
        logging.info('Using Genomic Dataset as Dataset')
    elif dataset_type == 'transfer_learning':
        dataset = GenomicTLDataset
        logging.info('Using Genomic Transfer Learning Dataset as Dataset')
    else:
        logging.error('Dataset type not found!')


    train_dataset = dataset(
        X_train, y_train, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    val_dataset = dataset(
        X_val, y_val, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    test_dataset = dataset(
        X_test, y_test, 
        app_config=app_config,
        resize_shape=resize_shape, 
    )

    return train_dataset, val_dataset, test_dataset


######################################### train dataloader end #########################################

######################################### infer dataloader start #########################################

def infer_dataset_details(
    task_data_path,
    app_config,
    skip_file_prefix = 'avg_', 
    save_file = False, 
    dataset_details_file = 'dataset_details.csv',
    only_genes = False
):

    chromosomes = os.listdir(task_data_path)
    
    logging.info('Available chromosomes: {}'.format(chromosomes))
    
    dataset_details = {
        'position': [], 
        'task_file_path': [],
        'chromosome_label': []
    }

    for chromosome in chromosomes:

        if chromosome in app_config['chromosomes']:
            files = os.listdir(os.path.join(task_data_path, chromosome))
            logging.info(f'chromosome: {chromosome} total files: {len(files)}')

            for file in files:
                if skip_file_prefix in file:
                    logging.info('skipped: {}'.format(file))
                    continue
                

                file_path = os.path.join(os.path.join(task_data_path, chromosome, file))
                chromosome_label = chromosome
                temp = file.split('/')[-1].split('.')[0].split('_')
                position = int(temp[-1])

                if only_genes:
                    if chromosome in gene_mapping.keys():
                        zones = gene_mapping[chromosome]
                        for zone_name, zone_data in zones.items():
                            zoom_start = zone_data['zoom_start']
                            zoom_end = zone_data['zoom_end']
                            if position >= zoom_start and position <= zoom_end:
                                dataset_details['task_file_path'].append(file_path)
                                dataset_details['chromosome_label'].append(chromosome_label)
                                dataset_details['position'].append(position)

                else:
                    dataset_details['task_file_path'].append(file_path)
                    dataset_details['chromosome_label'].append(chromosome_label)
                    dataset_details['position'].append(position)
        

    df_dataset_details = pd.DataFrame.from_dict(dataset_details)
    if save_file:
        df_dataset_details.to_csv(dataset_details_file, index=False)
    
    return df_dataset_details


def get_infer_data(app_config):
    
    seed = app_config['seed']

    task_3_data_path = app_config['task_3_data_path']
    task3_df = infer_dataset_details(task_3_data_path, app_config)
    task3_df = task3_df.rename(columns={"task_file_path": "task3_file_path"})

    task_2_data_path = app_config['task_2_data_path']
    task2_df = infer_dataset_details(task_2_data_path, app_config)
    task2_df = task2_df.rename(columns={"task_file_path": "task2_file_path"})

    task_1_data_path = app_config['task_1_data_path']
    task1_df = infer_dataset_details(task_1_data_path, app_config)
    task1_df = task1_df.rename(columns={"task_file_path": "task1_file_path"})

    all_task_df = task1_df.merge(
        task2_df, 
        on=['position', 'chromosome_label'], 
        how='outer', 
        suffixes=('_1', '_2')
    )

    all_task_df = all_task_df.merge(
        task3_df, 
        on=['position', 'chromosome_label'], 
        how='outer', 
        suffixes=('', '_3')
    )

    
    model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
    model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

    model_total_tasks_files = model_parallel_branches * model_per_branch_channels

    def window_task_df(task_path, task_file_path):
        task_data_path = app_config[task_path]
        task_df = infer_dataset_details(task_data_path, app_config)
        task_df = task_df.rename(columns={"task_file_path": task_file_path})
        return task_df

    # if we need extra files.... (for more channels or more branches...)
    if model_total_tasks_files == 9:
        # w2
        w2_task3_df = window_task_df('w2_task_3_data_path', 'w2_task3_file_path')
        all_task_df = all_task_df.merge(
            w2_task3_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_3')
        )

        w2_task2_df = window_task_df('w2_task_2_data_path', 'w2_task2_file_path')
        all_task_df = all_task_df.merge(
            w2_task2_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_2')
        )

        w2_task1_df = window_task_df('w2_task_1_data_path', 'w2_task1_file_path')
        all_task_df = all_task_df.merge(
            w2_task1_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_1')
        )

        # w3
        w3_task3_df = window_task_df('w3_task_3_data_path', 'w3_task3_file_path')
        all_task_df = all_task_df.merge(
            w3_task3_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_3')
        )

        w3_task2_df = window_task_df('w3_task_2_data_path', 'w3_task2_file_path')
        all_task_df = all_task_df.merge(
            w3_task2_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_2')
        )

        w3_task1_df = window_task_df('w3_task_1_data_path', 'w3_task1_file_path')
        all_task_df = all_task_df.merge(
            w3_task1_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_1')
        )
    
    logging.info('Infer dataset size: {}'.format(all_task_df.shape))

    return all_task_df


def get_infer_data_tl(app_config, only_genes=False):
    seed = app_config['seed']

    task_3_data_path = app_config['task_3_data_path']
    task3_df = infer_dataset_details(task_3_data_path, app_config, only_genes=only_genes)
    task3_df = task3_df.rename(columns={"task_file_path": "task3_file_path"})

    task_2_data_path = app_config['task_2_data_path']
    task2_df = infer_dataset_details(task_2_data_path, app_config,  only_genes=only_genes)
    task2_df = task2_df.rename(columns={"task_file_path": "task2_file_path"})

    task_1_data_path = app_config['task_1_data_path']
    task1_df = infer_dataset_details(task_1_data_path, app_config, only_genes=only_genes)
    task1_df = task1_df.rename(columns={"task_file_path": "task1_file_path"})

    all_task_df = task1_df.merge(
        task2_df, 
        on=['position', 'chromosome_label'], 
        how='outer', 
        suffixes=('_1', '_2')
    )

    all_task_df = all_task_df.merge(
        task3_df, 
        on=['position', 'chromosome_label'], 
        how='outer', 
        suffixes=('', '_3')
    )

    
    model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
    model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

    model_total_tasks_files = model_parallel_branches * model_per_branch_channels

    def window_task_df(task_path, task_file_path):
        task_data_path = app_config[task_path]
        task_df = infer_dataset_details(task_data_path, app_config, only_genes=only_genes)
        task_df = task_df.rename(columns={"task_file_path": task_file_path})
        return task_df

    # if we need extra files.... (for more channels or more branches...)
    if model_total_tasks_files == 27:
        # w2
        w2_task3_df = window_task_df('w2_task_3_data_path', 'w2_task3_file_path')
        all_task_df = all_task_df.merge(
            w2_task3_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_3')
        )

        w2_task2_df = window_task_df('w2_task_2_data_path', 'w2_task2_file_path')
        all_task_df = all_task_df.merge(
            w2_task2_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_2')
        )

        w2_task1_df = window_task_df('w2_task_1_data_path', 'w2_task1_file_path')
        all_task_df = all_task_df.merge(
            w2_task1_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w2_1')
        )

        # w3
        w3_task3_df = window_task_df('w3_task_3_data_path', 'w3_task3_file_path')
        all_task_df = all_task_df.merge(
            w3_task3_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_3')
        )

        w3_task2_df = window_task_df('w3_task_2_data_path', 'w3_task2_file_path')
        all_task_df = all_task_df.merge(
            w3_task2_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_2')
        )

        w3_task1_df = window_task_df('w3_task_1_data_path', 'w3_task1_file_path')
        all_task_df = all_task_df.merge(
            w3_task1_df, 
            on=['position', 'chromosome_label'], 
            how='outer', 
            suffixes=('', 'w3_1')
        )
    
    logging.info('Infer dataset size: {}'.format(all_task_df.shape))

    return all_task_df


def generate_infer_datasets(app_config, dataset_type='genomic', only_genes=False):

    # get data
    # X_test, y_test, X_train, y_train, X_val, y_val = get_infer_data(app_config)
    if dataset_type == 'genomic':
        infer_data = get_infer_data(app_config)
    elif dataset_type == 'transfer_learning':
        infer_data = get_infer_data_tl(app_config, only_genes=only_genes)

    dataset_ref_path = os.path.join(app_config['output_save_path'], 'dataset')
    os.makedirs(dataset_ref_path, exist_ok = True)

    ### save train, test, val dataset for ref
    # infer -------------------
    infer_save_path = os.path.join(dataset_ref_path, 'infer_data.csv')
    infer_data.to_csv(infer_save_path, index=False)
    #-------------------

    resize_shape = (32, 32)
    if 'resize_shape' in app_config:
        temp_r_shape = int(app_config['resize_shape'])
        resize_shape = (temp_r_shape, temp_r_shape)
    
    elif 'input_size' in cnn_model_metadata[app_config['cnn_model']]:
        temp_r_shape = int(cnn_model_metadata[app_config['cnn_model']]['input_size'])
        resize_shape = (temp_r_shape, temp_r_shape)
    
    else:
        logging.warning('using default resize shape')
    
    logging.info('resize shape used : {}'.format(resize_shape))

    dataset = None
    if dataset_type == 'genomic':
        dataset = ChromosomeDataset
        logging.info('Using Genomic Dataset as Dataset')
    elif dataset_type == 'transfer_learning':
        dataset = ChromosomeTLDataset
        logging.info('Using Genomic Transfer Learning Dataset as Dataset')
    else:
        logging.error('Dataset type not found!')

    infer_dataset = dataset(
        infer_data,
        app_config, 
        resize_shape = resize_shape, 
    )

    return infer_dataset


class ChromosomeDataset(Dataset):
    """Genomic Dataset """
    def __init__(self, x_dataframe, app_config, resize_shape=(32, 32), transform=None):
        """
        """
        self.app_config = app_config
        self.x_df = x_dataframe

        self.resize = resize_shape
        self.transform = transform

        self.model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
        self.model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels

        # self.model_parallel_channels = cnn_model_metadata[self.app_config['cnn_model']]['parallel_channels']
        self.transform_function_init()

        self.init_labels()


        # # self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        # if 'datasets' in app_config and app_config['datasets'] != '':
        #     self.labels = app_config['datasets']
        # else:
        #     self.labels = ['AI', 'BalSel', 'Neut', 'Sweeps']
        # # self.labels = ['Neut', 'BalSel', 'AI', 'Sweeps']
        # # self.le = LabelEncoder()
        # # self.le.fit(self.labels)

        # self.lb = LabelBinarizer()
        # self.lb.fit(self.labels) # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


    def __len__(self):
        return len(self.x_df)
    

    def __getitem__(self, idx):

        # print(idx)
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            logging.warning('index to list happened...')
        
        # ['task3_file_path', 'task2_file_path', 'task1_file_path']
        task_1_img_location = self.x_df.iloc[idx]['task1_file_path']
        task_2_img_location = self.x_df.iloc[idx]['task2_file_path']
        task_3_img_location = self.x_df.iloc[idx]['task3_file_path']

        # print(task_1_img_location)

        # label = self.le.transform([self.y_df.iloc[idx]])
        # label = self.lb.transform([self.y_df.iloc[idx]])
        position = self.x_df.iloc[idx]['position']
        chromosome_label = self.x_df.iloc[idx]['chromosome_label']
        
        # return task_1_img_location, task_2_img_location, task_3_img_location, label

        # load and resize the images
        ## task 1 data
        torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
        ## task 2 data
        torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
        ## task 3 data
        torch_t3_data = self.read_and_preprocess_data(task_3_img_location)
        # return torch_t2_data, torch_t3_data, label

        if self.model_total_tasks_files == 9:
            # w2
            task_1_img_location = self.x_df.iloc[idx]['w2_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w2_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w2_task3_file_path']

            # load and resize the images
            ## task 1 data
            w2_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            ## task 2 data
            w2_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            ## task 3 data
            w2_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            # w3
            task_1_img_location = self.x_df.iloc[idx]['w3_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w3_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w3_task3_file_path']

            # load and resize the images
            ## task 1 data
            w3_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            ## task 2 data
            w3_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            ## task 3 data
            w3_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)
        

        # add channels for necessary configs models
        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            torch_t1_data = torch.cat((torch_t1_data, w2_torch_t1_data, w3_torch_t1_data), 0)
            torch_t2_data = torch.cat((torch_t2_data, w2_torch_t2_data, w3_torch_t2_data), 0)
            torch_t3_data = torch.cat((torch_t3_data, w2_torch_t3_data, w3_torch_t3_data), 0)

        # transform data
        if self.transform:

            if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
                torch_t1_data = self.transform(torch_t1_data, window=1, task=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, window=1, task=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, window=1, task=3, resize_shape=self.resize)

            elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:
                torch_t1_data = self.transform(torch_t1_data, window=1, task=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, window=1, task=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, window=1, task=3, resize_shape=self.resize)

                w2_torch_t1_data = self.transform(w2_torch_t1_data, window=2, task=1, resize_shape=self.resize)
                w2_torch_t2_data = self.transform(w2_torch_t2_data, window=2, task=2, resize_shape=self.resize)
                w2_torch_t3_data = self.transform(w2_torch_t3_data, window=2, task=3, resize_shape=self.resize)

                w3_torch_t1_data = self.transform(w3_torch_t1_data, window=3, task=1, resize_shape=self.resize)
                w3_torch_t2_data = self.transform(w3_torch_t2_data, window=3, task=2, resize_shape=self.resize)
                w3_torch_t3_data = self.transform(w3_torch_t3_data, window=3, task=3, resize_shape=self.resize)
            
            elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                torch_t1_data = self.transform(torch_t1_data, branch=1, resize_shape=self.resize)
                torch_t2_data = self.transform(torch_t2_data, branch=2, resize_shape=self.resize)
                torch_t3_data = self.transform(torch_t3_data, branch=3, resize_shape=self.resize)
            
            else:
                logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                    self.model_parallel_branches, self.model_per_branch_channels))
        
        
        # return data
        # return torch_t1_data, torch_t2_data, torch_t3_data, position, chromosome_label
        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
            return torch_t1_data, torch_t2_data, torch_t3_data, position, chromosome_label
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:
            return torch_t1_data, torch_t2_data, torch_t3_data, \
                w2_torch_t1_data, w2_torch_t2_data, w2_torch_t3_data, \
                w3_torch_t1_data, w3_torch_t2_data, w3_torch_t3_data, \
                position, chromosome_label
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            return torch_t1_data, torch_t2_data, torch_t3_data, position, chromosome_label
        
        else:
            logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                self.model_parallel_branches, self.model_per_branch_channels))
            return -1


    def transform_function_init(self):
        # app config
        cnn_model = self.app_config['cnn_model']
        cnn_model_config = cnn_model_metadata[cnn_model]
        input_size = cnn_model_config['input_size']

        # find dataset metadata for means and stds
        if 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            self.dataset_name = self.app_config['dataset_name']
        else:
            self.dataset_name = 'CEU6000'
        logging.info(f'Using {self.dataset_name} dataset for dataset mean and std.')

        ds_metadata = dataset_metadata[self.dataset_name]['dataset_metadata']


        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                w1_means_stds['means'], 
                w1_means_stds['means'], 
                w1_means_stds['means'],
            ]
            self.stds = [
                [-1], 
                w1_means_stds['stds'], 
                w1_means_stds['stds'], 
                w1_means_stds['stds'],
            ]
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:

            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]
            
            window_size = self.app_config['w2_window_size']
            # input_size = 32
            stride = 10
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            # input_size = 32
            stride = 10
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                w1_means_stds['means'], 
                w2_means_stds['means'], 
                w3_means_stds['means'],
            ]
            self.stds = [
                [-1], 
                w1_means_stds['stds'], 
                w2_means_stds['stds'], 
                w3_means_stds['stds'],
            ]
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w2_window_size']
            # input_size = 32
            stride = 10
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            # input_size = 32
            stride = 10
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]


            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w2_means_stds['means'][1], w3_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w2_means_stds['means'][2], w3_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w2_means_stds['means'][3], w3_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w2_means_stds['stds'][1], w3_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w2_means_stds['stds'][2], w3_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w2_means_stds['stds'][3], w3_means_stds['stds'][3]], (-1, 1, 1)), 
            ]

        else:
            logging.error('No code for model_parallel_branches {} and model_per_branch_channels {}'.format(self.self.model_parallel_branches, self.model_per_branch_channels))


        transform_func = None
        if self.app_config['normalization'] != '':
            # normalization: 'image_wise_standardization'
            if self.app_config['normalization'] == 'image_wise_standardization':
                transform_func = self.transform_image_wise_standardization
                logging.info('using transform_image_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'image_wise_normalizaton':
                transform_func = self.transform_image_wise_normalization
                logging.info('using transform_image_wise_normalization function for data normalization process')

            elif self.app_config['normalization'] == 'dataset_wise_standardization':

                # for br3, ch3 => we need to select different one
                if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                    transform_func = self.transform_3channel_3branch_standardization
                else:
                    transform_func = self.transform_dataset_wise_standardization
                logging.info('using transform_dataset_wise_standardization function for data normalization process')
            
            elif self.app_config['normalization'] == 'histogram_equalization':
                if self.model_per_branch_channels == 1:
                    transform_func = self.transform_histogram_equalization
                    logging.info('using transform_histogram_equalization function for data normalization process')
                else:
                    logging.error(f'no function description for {self.app_config["normalization"]}')
                    logging.info('Using no normalization in data')
            
            elif self.app_config['normalization'] == 'domain_adaptation':
                transform_func = self.transform_domain_adaptation
                logging.info('using transform_domain_adaptation function for data normalization process')

            else:
                logging.warning(f'no function description for {self.app_config["normalization"]}')
                logging.info('Using no normalization in data')

        self.transform = transform_func


    def transform_3channel_3branch_standardization(self, data, **kwargs):
        """
        """

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'branch' in kwargs:
            logging.error('expected branch value of 0/1/2!!')

        # task = int(kwargs['task'])
        branch = int(kwargs['branch'])
        # standardized data
        s_data = (data - self.means[branch]) / (self.stds[branch])

        return s_data.float()


    def transform_dataset_wise_standardization(self, data, **kwargs):
        """
        https://www.geeksforgeeks.org/args-kwargs-python/
        """
        # logging.info('means: {}'.format(means))

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'task' in kwargs:
            logging.error('expected task value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        task = int(kwargs['task'])
        window = int(kwargs['window'])
        # standardized data
        s_data = (data - self.means[window][task]) / (self.stds[window][task])

        return s_data


    def transform_image_wise_standardization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate mean and std
        # https://pytorch.org/docs/stable/generated/torch.std_mean.html
        # dim 2, 3 are 32x32 image dim in batch
        # t_std, t_mean  = torch.std_mean(data, dim=(2,3), keepdim=True)
        # single image case, dim 1, 2 are 32x32 image dim
        t_std, t_mean  = torch.std_mean(data, dim=(1,2), keepdim=True)

        # normalize data
        s_data = (data - t_mean) / t_std

        return s_data


    def transform_image_wise_normalization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate min and max
        t_min, t_max  = data.min(), data.max()

        # normalize data
        s_data = (data - t_min) / (t_max - t_min)

        return s_data


    def transform_histogram_equalization(self, data, **kwargs):
        """
        """
        def normalize8(img):
            mn = img.min()
            mx = img.max()
            mx -= mn
            img = ((img - mn)/ mx) * 255
            return img.astype(np.uint8)

        # convert to numpy 
        data = data.squeeze().cpu().detach().numpy() # (32, 32)
        task = int(kwargs['task'])
        if task == 1:
            data = data.astype(np.uint8)
        else:
            data = normalize8(data)
        
        data = cv2.equalizeHist(data) / 255.0

        # convert to torch
        data = self.convert_numpy_to_torch(data) # (1, 32, 32)
        return data


    def transform_domain_adaptation(self, data, **kwargs):
        # make channel mean zero and std one
        data = self.transform_image_wise_standardization(data)

        # make channel mean and std same as simulation data
        if not torch.is_tensor(data):
            logging.error('expected torch data!')

        if 'task' in kwargs:
            task = int(kwargs['task'])
            window = int(kwargs['window'])
            # standardized data
            s_data = (data * (self.stds[window][task])) + self.means[window][task]

        elif 'branch' in kwargs:
            
            branch = int(kwargs['branch'])
            # standardized data
            s_data = (data * (self.stds[branch])) + self.means[branch]
            s_data = s_data.float()
        
        else:
            logging.error('Not implemented domain adaptation function')

        return s_data


    def read_and_preprocess_data(self, task_img_location, dtype=None):
        task_data = pd.read_csv(task_img_location)
        if dtype is not None:
            task_data = task_data.to_numpy(dtype=dtype)
        else:
            task_data = task_data.to_numpy()
        # print(type(data_2), data_2.shape)

        # # add condition for none resize
        # task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
        # torch_task_data = torch.from_numpy(task_data)
        # torch_task_data = torch_task_data.type(dtype=torch.float32)
        # torch_task_data = torch_task_data.unsqueeze(0)
        # return torch_task_data

        # add condition for none resize
        if self.resize is not None:
            # interpolation ref
            # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
            if 'resize_interpolation' in self.app_config and self.app_config['resize_interpolation'] != '':
                if self.app_config['resize_interpolation'] == 'same':
                    task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                elif self.app_config['resize_interpolation'] == 'shape_dependent':
                    if task_data.shape[0] > self.resize[0]:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                    else:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
            else:
                task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
        
        # torch_task_data = torch.from_numpy(task_data)
        # torch_task_data = torch_task_data.type(dtype=torch.float32)
        # torch_task_data = torch_task_data.unsqueeze(0)
        torch_task_data = self.convert_numpy_to_torch(task_data=task_data)

        return torch_task_data

    
    def convert_numpy_to_torch(self, task_data):
        torch_task_data = torch.from_numpy(task_data)
        torch_task_data = torch_task_data.type(dtype=torch.float32)
        torch_task_data = torch_task_data.unsqueeze(0)

        return torch_task_data

    
    def init_labels(self):
        if 'datasets' in self.app_config and self.app_config['datasets'] != '':
            self.labels = self.app_config['datasets']
        
        elif 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            if self.app_config['dataset_name'] in dataset_metadata.keys():
                self.labels = dataset_metadata[self.app_config['dataset_name']]['labels']
            else:
                self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        else:
            self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']

        logging.info(f'Using labels: {self.labels}')

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)    # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


class ChromosomeTLDataset(Dataset):
    """Genomic Dataset """
    def __init__(self, x_dataframe, app_config, resize_shape=(299, 299), transform=None):
        """
        """
        self.app_config = app_config
        self.x_df = x_dataframe

        self.resize = resize_shape
        self.transform = transform

        self.model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
        self.model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels

        self.transform_function_init()

        self.init_labels()


    def __len__(self):
        return len(self.x_df)
    

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
            logging.warning('index to list happened...')
        
        task_1_img_location = self.x_df.iloc[idx]['task1_file_path']
        task_2_img_location = self.x_df.iloc[idx]['task2_file_path']
        task_3_img_location = self.x_df.iloc[idx]['task3_file_path']

        position = self.x_df.iloc[idx]['position']
        chromosome_label = self.x_df.iloc[idx]['chromosome_label']
        

        # load and resize the images
        torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
        torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
        torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

        # copy same data from channels for each branch
        torch_t1_data = torch.cat((torch_t1_data, torch_t1_data, torch_t1_data), 0)
        torch_t2_data = torch.cat((torch_t2_data, torch_t2_data, torch_t2_data), 0)
        torch_t3_data = torch.cat((torch_t3_data, torch_t3_data, torch_t3_data), 0)

        if self.model_total_tasks_files == 27:
            # w2
            task_1_img_location = self.x_df.iloc[idx]['w2_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w2_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w2_task3_file_path']

            # load and resize the images
            w2_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            w2_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            w2_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            w2_torch_t1_data = torch.cat((w2_torch_t1_data, w2_torch_t1_data, w2_torch_t1_data), 0)
            w2_torch_t2_data = torch.cat((w2_torch_t2_data, w2_torch_t2_data, w2_torch_t2_data), 0)
            w2_torch_t3_data = torch.cat((w2_torch_t3_data, w2_torch_t3_data, w2_torch_t3_data), 0)

            # w3
            task_1_img_location = self.x_df.iloc[idx]['w3_task1_file_path']
            task_2_img_location = self.x_df.iloc[idx]['w3_task2_file_path']
            task_3_img_location = self.x_df.iloc[idx]['w3_task3_file_path']

            # load and resize the images
            w3_torch_t1_data = self.read_and_preprocess_data(task_1_img_location, dtype='uint8')
            w3_torch_t2_data = self.read_and_preprocess_data(task_2_img_location)
            w3_torch_t3_data = self.read_and_preprocess_data(task_3_img_location)

            w3_torch_t1_data = torch.cat((w3_torch_t1_data, w3_torch_t1_data, w3_torch_t1_data), 0)
            w3_torch_t2_data = torch.cat((w3_torch_t2_data, w3_torch_t2_data, w3_torch_t2_data), 0)
            w3_torch_t3_data = torch.cat((w3_torch_t3_data, w3_torch_t3_data, w3_torch_t3_data), 0)
        

        # transform data
        if self.transform:

            torch_t1_data = self.transform(torch_t1_data, branch=1, resize_shape=self.resize)
            torch_t2_data = self.transform(torch_t2_data, branch=2, resize_shape=self.resize)
            torch_t3_data = self.transform(torch_t3_data, branch=3, resize_shape=self.resize)

            if self.model_total_tasks_files == 27:

                w2_torch_t1_data = self.transform(w2_torch_t1_data, branch=4, resize_shape=self.resize)
                w2_torch_t2_data = self.transform(w2_torch_t2_data, branch=5, resize_shape=self.resize)
                w2_torch_t3_data = self.transform(w2_torch_t3_data, branch=6, resize_shape=self.resize)

                w3_torch_t1_data = self.transform(w3_torch_t1_data, branch=7, resize_shape=self.resize)
                w3_torch_t2_data = self.transform(w3_torch_t2_data, branch=8, resize_shape=self.resize)
                w3_torch_t3_data = self.transform(w3_torch_t3_data, branch=9, resize_shape=self.resize)
        
        
        if self.model_total_tasks_files == 27:
            return torch_t1_data, torch_t2_data, torch_t3_data, \
                w2_torch_t1_data, w2_torch_t2_data, w2_torch_t3_data, \
                w3_torch_t1_data, w3_torch_t2_data, w3_torch_t3_data, \
                position, chromosome_label
        else:
            return torch_t1_data, torch_t2_data, torch_t3_data, position, chromosome_label



    def transform_function_init(self):
        # app config
        cnn_model = self.app_config['cnn_model']
        cnn_model_config = cnn_model_metadata[cnn_model]
        input_size = cnn_model_config['input_size']

        # find dataset metadata for means and stds
        if 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            self.dataset_name = self.app_config['dataset_name']
        else:
            self.dataset_name = 'CEU6000'
        logging.info(f'Using {self.dataset_name} dataset for dataset mean and std.')

        ds_metadata = dataset_metadata[self.dataset_name]['dataset_metadata']


        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]


            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w1_means_stds['means'][1], w1_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w1_means_stds['means'][2], w1_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w1_means_stds['means'][3], w1_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w1_means_stds['stds'][1], w1_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w1_means_stds['stds'][2], w1_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w1_means_stds['stds'][3], w1_means_stds['stds'][3]], (-1, 1, 1)), 
            ]
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 3:
            window_size = self.app_config['w1_window_size']
            # input_size = 32
            stride = 10
            w1_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w2_window_size']
            w2_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            window_size = self.app_config['w3_window_size']
            w3_means_stds = ds_metadata['window_size'][window_size]['input_size'][input_size]['stride'][stride]

            self.means = [
                [-1], 
                np.reshape([w1_means_stds['means'][1], w1_means_stds['means'][1], w1_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][2], w1_means_stds['means'][2], w1_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['means'][3], w1_means_stds['means'][3], w1_means_stds['means'][3]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][1], w2_means_stds['means'][1], w2_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][2], w2_means_stds['means'][2], w2_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['means'][3], w2_means_stds['means'][3], w2_means_stds['means'][3]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][1], w3_means_stds['means'][1], w3_means_stds['means'][1]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][2], w3_means_stds['means'][2], w3_means_stds['means'][2]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['means'][3], w3_means_stds['means'][3], w3_means_stds['means'][3]], (-1, 1, 1)), 
            ]
            self.stds = [
                [-1], 
                np.reshape([w1_means_stds['stds'][1], w1_means_stds['stds'][1], w1_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][2], w1_means_stds['stds'][2], w1_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w1_means_stds['stds'][3], w1_means_stds['stds'][3], w1_means_stds['stds'][3]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][1], w2_means_stds['stds'][1], w2_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][2], w2_means_stds['stds'][2], w2_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w2_means_stds['stds'][3], w2_means_stds['stds'][3], w2_means_stds['stds'][3]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][1], w3_means_stds['stds'][1], w3_means_stds['stds'][1]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][2], w3_means_stds['stds'][2], w3_means_stds['stds'][2]], (-1, 1, 1)), 
                np.reshape([w3_means_stds['stds'][3], w3_means_stds['stds'][3], w3_means_stds['stds'][3]], (-1, 1, 1)), 
            ]
        
        else:
            logging.error('No code for model_parallel_branches {} and model_per_branch_channels {}'.format(self.self.model_parallel_branches, self.model_per_branch_channels))



        transform_func = None
        transform_func = None
        if self.app_config['normalization'] != '':
            # normalization: 'image_wise_standardization'
            if self.app_config['normalization'] == 'image_wise_standardization':
                #TODO
                transform_func = self.transform_image_wise_standardization
                logging.info('using transform_image_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'image_wise_normalizaton':
                #TODO
                transform_func = self.transform_image_wise_normalization
                logging.info('using transform_image_wise_normalization function for data normalization process')

            elif self.app_config['normalization'] == 'dataset_wise_standardization':

                # for br3, ch3 => we need to select different one
                # if self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
                #     transform_func = self.transform_3channel_3branch_standardization
                # else:
                #     transform_func = self.transform_dataset_wise_standardization
                transform_func = self.transform_3channel_3branch_standardization
                logging.info('using transform_dataset_wise_standardization function for data normalization process')

            elif self.app_config['normalization'] == 'histogram_equalization':
                #TODO
                if self.model_per_branch_channels == 1:
                    transform_func = self.transform_histogram_equalization
                else:
                    logging.error(f'no function description for {self.app_config["normalization"]}')
                    logging.info('Using no normalization in data')

            else:
                logging.warning(f'no function description for {self.app_config["normalization"]}')
                logging.info('Using no normalization in data')

        self.transform = transform_func


    def transform_3channel_3branch_standardization(self, data, **kwargs):
        """
        """

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'branch' in kwargs:
            logging.error('expected branch value of 0/1/2!!')

        # task = int(kwargs['task'])
        branch = int(kwargs['branch'])
        # standardized data
        s_data = (data - self.means[branch]) / (self.stds[branch])

        return s_data.float()


    def transform_dataset_wise_standardization(self, data, **kwargs):
        """
        https://www.geeksforgeeks.org/args-kwargs-python/
        """
        # logging.info('means: {}'.format(means))

        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        if not 'task' in kwargs:
            logging.error('expected task value of 0/1/2!!')

        # if kwargs['task'] == 1:
        #     # divide by 32( max value in training dataset )
        #     data = data / 32

        task = int(kwargs['task'])
        window = int(kwargs['window'])
        # standardized data
        s_data = (data - self.means[window][task]) / (self.stds[window][task])

        return s_data


    def transform_image_wise_standardization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate mean and std
        # https://pytorch.org/docs/stable/generated/torch.std_mean.html
        # dim 2, 3 are 32x32 image dim in batch
        # t_std, t_mean  = torch.std_mean(data, dim=(2,3), keepdim=True)
        # single image case, dim 1, 2 are 32x32 image dim
        t_std, t_mean  = torch.std_mean(data, dim=(1,2), keepdim=True)

        # normalize data
        s_data = (data - t_mean) / t_std

        return s_data


    def transform_image_wise_normalization(self, data, **kwargs):
        """
        """
        if not torch.is_tensor(data):
            logging.error('expected torch data!')
        
        # calculate min and max
        t_min, t_max  = data.min(), data.max()

        # normalize data
        s_data = (data - t_min) / (t_max - t_min)

        return s_data


    def transform_histogram_equalization(self, data, **kwargs):
        """
        """
        def normalize8(img):
            mn = img.min()
            mx = img.max()
            mx -= mn
            img = ((img - mn)/ mx) * 255
            return img.astype(np.uint8)

        # convert to numpy 
        data = data.squeeze().cpu().detach().numpy() # (32, 32)
        task = int(kwargs['task'])
        if task == 1:
            data = data.astype(np.uint8)
        else:
            data = normalize8(data)
        
        data = cv2.equalizeHist(data) / 255.0

        # convert to torch
        data = self.convert_numpy_to_torch(data) # (1, 32, 32)
        return data


    def transform_domain_adaptation(self, data, **kwargs):
        # make channel mean zero and std one
        data = self.transform_image_wise_standardization(data)

        # make channel mean and std same as simulation data
        if not torch.is_tensor(data):
            logging.error('expected torch data!')

        if 'task' in kwargs:
            task = int(kwargs['task'])
            window = int(kwargs['window'])
            # standardized data
            s_data = (data * (self.stds[window][task])) + self.means[window][task]

        elif 'branch' in kwargs:
            
            branch = int(kwargs['branch'])
            # standardized data
            s_data = (data * (self.stds[branch])) + self.means[branch]
            s_data = s_data.float()
        
        else:
            logging.error('Not implemented domain adaptation function')

        return s_data


    def read_and_preprocess_data(self, task_img_location, dtype=None):
        task_data = pd.read_csv(task_img_location)
        if dtype is not None:
            task_data = task_data.to_numpy(dtype=dtype)
        else:
            task_data = task_data.to_numpy()
        # print(type(data_2), data_2.shape)

        # add condition for none resize
        if self.resize is not None:
            # interpolation ref
            # https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html#ga47a974309e9102f5f08231edc7e7529d
            if 'resize_interpolation' in self.app_config and self.app_config['resize_interpolation'] != '':
                if self.app_config['resize_interpolation'] == 'same':
                    task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                elif self.app_config['resize_interpolation'] == 'shape_dependent':
                    if task_data.shape[0] > self.resize[0]:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
                    else:
                        task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_LINEAR)
            else:
                task_data = cv2.resize(task_data, dsize=self.resize, interpolation=cv2.INTER_AREA)
        
        torch_task_data = self.convert_numpy_to_torch(task_data=task_data)

        return torch_task_data

    
    def convert_numpy_to_torch(self, task_data):
        torch_task_data = torch.from_numpy(task_data)
        torch_task_data = torch_task_data.type(dtype=torch.float32)
        torch_task_data = torch_task_data.unsqueeze(0)

        return torch_task_data

    
    def init_labels(self):
        if 'datasets' in self.app_config and self.app_config['datasets'] != '':
            self.labels = self.app_config['datasets']
        
        elif 'dataset_name' in self.app_config and self.app_config['dataset_name'] != '':
            if self.app_config['dataset_name'] in dataset_metadata.keys():
                self.labels = dataset_metadata[self.app_config['dataset_name']]['labels']
            else:
                self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']
        else:
            self.labels = ['AI', 'Sweeps', 'BalSel', 'Neut']

        logging.info(f'Using labels: {self.labels}')

        self.lb = LabelBinarizer()
        self.lb.fit(self.labels)    # labels are in this order: ['AI', 'BalSel', 'Neut', 'Sweeps']
        # self.lb.classes_ = self.labels # to fit the classes as we have in our labels


######################################### infer dataloader end #########################################

######################################### util functions start #########################################
def mean_stds_br9_ch1(trainloader):
    
    task_1 = None
    task_2 = None
    task_3 = None


    w2_task_1 = None
    w2_task_2 = None
    w2_task_3 = None


    w3_task_1 = None
    w3_task_2 = None
    w3_task_3 = None


    for i, data in enumerate(trainloader, 0):
        
        if task_1 is None:
            task_1 = data[0].numpy()
        else:
            task_1 = np.append(task_1, data[0].numpy(), axis=0)
        
        if task_2 is None:
            task_2 = data[1].numpy()
        else:
            task_2 = np.append(task_2, data[1].numpy(), axis=0)
        
        if task_3 is None:
            task_3 = data[2].numpy()
        else:
            task_3 = np.append(task_3, data[2].numpy(), axis=0)
        
        # w2
        if w2_task_1 is None:
            w2_task_1 = data[3].numpy()
        else:
            w2_task_1 = np.append(w2_task_1, data[3].numpy(), axis=0)
        
        if w2_task_2 is None:
            w2_task_2 = data[4].numpy()
        else:
            w2_task_2 = np.append(w2_task_2, data[4].numpy(), axis=0)
        
        if w2_task_3 is None:
            w2_task_3 = data[5].numpy()
        else:
            w2_task_3 = np.append(w2_task_3, data[5].numpy(), axis=0)
        
        # w3
        if w3_task_1 is None:
            w3_task_1 = data[6].numpy()
        else:
            w3_task_1 = np.append(w3_task_1, data[6].numpy(), axis=0)
        
        if w3_task_2 is None:
            w3_task_2 = data[7].numpy()
        else:
            w3_task_2 = np.append(w3_task_2, data[7].numpy(), axis=0)
        
        if w3_task_3 is None:
            w3_task_3 = data[8].numpy()
        else:
            w3_task_3 = np.append(w3_task_3, data[8].numpy(), axis=0)

        if i % 100 == 99:
            logging.info(f'done: [{i+1} / {len(trainloader)}]')


    # min max calculation
    logging.info('======================================')
    logging.info('w1 min and max:')
    logging.info(f'{task_1.min()}, {task_1.max()}')
    logging.info(f'{task_2.min()}, {task_2.max()}')
    logging.info(f'{task_3.min()}, {task_3.max()}')

    logging.info('w2 min and max:')
    logging.info(f'{w2_task_1.min()}, {w2_task_1.max()}')
    logging.info(f'{w2_task_2.min()}, {w2_task_2.max()}')
    logging.info(f'{w2_task_3.min()}, {w2_task_3.max()}')

    logging.info('w3 min and max:')
    logging.info(f'{w3_task_1.min()}, {w3_task_1.max()}')
    logging.info(f'{w3_task_2.min()}, {w3_task_2.max()}')
    logging.info(f'{w3_task_3.min()}, {w3_task_3.max()}')
    logging.info('======================================')

    # mean and std calculation
    logging.info('w1 mean and std:')
    # task_1 = x[0].numpy()
    t1_mean = task_1.mean(axis=(0,2,3))
    t1_std = task_1.std(axis=(0,2,3))
    logging.info(f'{t1_mean}, {t1_std}')

    # task_2 = x[1].numpy()
    t2_mean = task_2.mean(axis=(0,2,3))
    t2_std = task_2.std(axis=(0,2,3))
    logging.info(f'{t2_mean}, {t2_std}')

    # task_3 = x[2].numpy()
    t3_mean = task_3.mean(axis=(0,2,3))
    t3_std = task_3.std(axis=(0,2,3))
    logging.info(f'{t3_mean}, {t3_std}')

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')

    logging.info('w2 mean and std:')
    # task_1 = x[0].numpy()
    w2_t1_mean = w2_task_1.mean(axis=(0,2,3))
    w2_t1_std = w2_task_1.std(axis=(0,2,3))
    logging.info(f'{w2_t1_mean}, {w2_t1_std}')

    # task_2 = x[1].numpy()
    w2_t2_mean = w2_task_2.mean(axis=(0,2,3))
    w2_t2_std = w2_task_2.std(axis=(0,2,3))
    logging.info(f'{w2_t2_mean}, {w2_t2_std}')

    # task_3 = x[2].numpy()
    w2_t3_mean = w2_task_3.mean(axis=(0,2,3))
    w2_t3_std = w2_task_3.std(axis=(0,2,3))
    logging.info(f'{w2_t3_mean}, {w2_t3_std}')

    logging.info(f'means: {w2_t1_mean}, {w2_t2_mean}, {w2_t3_mean}')
    logging.info(f'stds: {w2_t1_std}, {w2_t2_std}, {w2_t3_std}')
    logging.info('======================================')

    logging.info('w3 mean and std:')
    # task_1 = x[0].numpy()
    w3_t1_mean = w3_task_1.mean(axis=(0,2,3))
    w3_t1_std = w3_task_1.std(axis=(0,2,3))
    logging.info(f'{w3_t1_mean}, {w3_t1_std}')

    # task_2 = x[1].numpy()
    w3_t2_mean = w3_task_2.mean(axis=(0,2,3))
    w3_t2_std = w3_task_2.std(axis=(0,2,3))
    logging.info(f'{w3_t2_mean}, {w3_t2_std}')

    # task_3 = x[2].numpy()
    w3_t3_mean = w3_task_3.mean(axis=(0,2,3))
    w3_t3_std = w3_task_3.std(axis=(0,2,3))
    logging.info(f'{w3_t3_mean}, {w3_t3_std}')

    logging.info(f'means: {w3_t1_mean}, {w3_t2_mean}, {w3_t3_mean}')
    logging.info(f'stds: {w3_t1_std}, {w3_t2_std}, {w3_t3_std}')
    logging.info('======================================')


def mean_stds_br3_ch1(trainloader):
    
    task_1 = None
    task_2 = None
    task_3 = None


    for i, data in enumerate(trainloader, 0):
        
        if task_1 is None:
            task_1 = data[0].numpy()
        else:
            task_1 = np.append(task_1, data[0].numpy(), axis=0)
        
        if task_2 is None:
            task_2 = data[1].numpy()
        else:
            task_2 = np.append(task_2, data[1].numpy(), axis=0)
        
        if task_3 is None:
            task_3 = data[2].numpy()
        else:
            task_3 = np.append(task_3, data[2].numpy(), axis=0)
        
        
        if i % 100 == 99:
            logging.info(f'done: [{i+1} / {len(trainloader)}]')



    # print('======================================')
    # print('w1 min and max:')
    # print(task_1.min(), task_1.max())
    # print(task_2.min(), task_2.max())
    # print(task_3.min(), task_3.max())
    
    logging.info('======================================')
    logging.info('w1 min and max:')
    logging.info(f'{task_1.min()}, {task_1.max()}')
    logging.info(f'{task_2.min()}, {task_2.max()}')
    logging.info(f'{task_3.min()}, {task_3.max()}')
    
    logging.info('======================================')
    logging.info('w1 mean and std:')
    # task_1 = x[0].numpy()
    t1_mean = task_1.mean(axis=(0,2,3))
    t1_std = task_1.std(axis=(0,2,3))
    logging.info(f'{t1_mean}, {t1_std}')

    # task_2 = x[1].numpy()
    t2_mean = task_2.mean(axis=(0,2,3))
    t2_std = task_2.std(axis=(0,2,3))
    logging.info(f'{t2_mean}, {t2_std}')

    # task_3 = x[2].numpy()
    t3_mean = task_3.mean(axis=(0,2,3))
    t3_std = task_3.std(axis=(0,2,3))
    logging.info(f'{t3_mean}, {t3_std}')

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')


def mean_stds_br3_ch1_mem_eff(trainloader):

    logging.info('Calculating mean')
    
    task_1 = None
    task_2 = None
    task_3 = None


    for i, data in enumerate(trainloader, 0):
        
        if task_1 is None:
            task_1 = np.sum(data[0].numpy(), axis=0) / data[0].shape[0]
        else:
            task_1 = np.add(task_1, np.sum(data[0].numpy(), axis=0) / data[0].shape[0])
        
        if task_2 is None:
            # task_2 = data[1].numpy()
            task_2 = np.sum(data[1].numpy(), axis=0) / data[1].shape[0]
        else:
            # task_2 = np.append(task_2, data[1].numpy(), axis=0)
            task_2 = np.add(task_2, np.sum(data[1].numpy(), axis=0) / data[1].shape[0])
        
        if task_3 is None:
            # task_3 = data[2].numpy()
            task_3 = np.sum(data[2].numpy(), axis=0) / data[2].shape[0]
        else:
            # task_3 = np.append(task_3, data[2].numpy(), axis=0)
            task_3 = np.add(task_3, np.sum(data[2].numpy(), axis=0) / data[2].shape[0])
        
        
        if i % 100 == 99:
            logging.info(f'done: [{i+1} / {len(trainloader)}]')


    task_1 = task_1 / (i+1)
    task_2 = task_2 / (i+1)
    task_3 = task_3 / (i+1)
    

    logging.info('======================================')
    logging.info('w1 mean')
    # task_1 = x[0].numpy()
    t1_mean = task_1.mean(axis=(1,2))
    # t1_std = task_1.std(axis=(1,2))
    # logging.info(f'{t1_mean}, {t1_std}')

    # task_2 = x[1].numpy()
    t2_mean = task_2.mean(axis=(1,2))
    # t2_std = task_2.std(axis=(1,2))
    # logging.info(f'{t2_mean}, {t2_std}')

    # task_3 = x[2].numpy()
    t3_mean = task_3.mean(axis=(1,2))
    # t3_std = task_3.std(axis=(1,2))
    # logging.info(f'{t3_mean}, {t3_std}')

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    # logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')


    logging.info('======================================')
    logging.info('Calculating std')


    def var(a, mean):

        # N = len(a)
        N = 1
        for i in a.shape:
            N *= i

        d2 = (a - mean)**2
        var = d2.sum() / (N)

        return var


    vars_t1 = 0.0
    vars_t2 = 0.0
    vars_t3 = 0.0

    for i, data in enumerate(trainloader, 0):
        # eff way
        vars_t1 += var(data[0].numpy(), t1_mean[0])
        vars_t2 += var(data[1].numpy(), t2_mean[0])
        vars_t3 += var(data[2].numpy(), t3_mean[0])

        
        if i % 100 == 99:
            print(f'done: [{i+1} / {len(trainloader)}]')
    
    
    logging.info('w1 std')
    t1_std = (vars_t1 / (i+1))**0.5
    t2_std = (vars_t2 / (i+1))**0.5
    t3_std = (vars_t3 / (i+1))**0.5

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')

    
def mean_stds_br3_ch1_mem_eff_w2(trainloader):

    logging.info('Calculating w2 mean')
    
    task_1 = None
    task_2 = None
    task_3 = None


    for i, data in enumerate(trainloader, 0):
        
        if task_1 is None:
            task_1 = np.sum(data[3].numpy(), axis=0) / data[3].shape[0]
        else:
            task_1 = np.add(task_1, np.sum(data[3].numpy(), axis=0) / data[3].shape[0])
        
        if task_2 is None:
            # task_2 = data[1].numpy()
            task_2 = np.sum(data[4].numpy(), axis=0) / data[4].shape[0]
        else:
            # task_2 = np.append(task_2, data[1].numpy(), axis=0)
            task_2 = np.add(task_2, np.sum(data[4].numpy(), axis=0) / data[4].shape[0])
        
        if task_3 is None:
            # task_3 = data[2].numpy()
            task_3 = np.sum(data[5].numpy(), axis=0) / data[5].shape[0]
        else:
            # task_3 = np.append(task_3, data[2].numpy(), axis=0)
            task_3 = np.add(task_3, np.sum(data[5].numpy(), axis=0) / data[5].shape[0])
        
        
        if i % 100 == 99:
            logging.info(f'done: [{i+1} / {len(trainloader)}]')


    task_1 = task_1 / (i+1)
    task_2 = task_2 / (i+1)
    task_3 = task_3 / (i+1)
    

    logging.info('======================================')
    logging.info('w2 mean')
    # task_1 = x[0].numpy()
    t1_mean = task_1.mean(axis=(1,2))
    # t1_std = task_1.std(axis=(1,2))
    # logging.info(f'{t1_mean}, {t1_std}')

    # task_2 = x[1].numpy()
    t2_mean = task_2.mean(axis=(1,2))
    # t2_std = task_2.std(axis=(1,2))
    # logging.info(f'{t2_mean}, {t2_std}')

    # task_3 = x[2].numpy()
    t3_mean = task_3.mean(axis=(1,2))
    # t3_std = task_3.std(axis=(1,2))
    # logging.info(f'{t3_mean}, {t3_std}')

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    # logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')


    logging.info('======================================')
    logging.info('Calculating std')


    def var(a, mean):

        # N = len(a)
        N = 1
        for i in a.shape:
            N *= i

        d2 = (a - mean)**2
        var = d2.sum() / (N)

        return var


    vars_t1 = 0.0
    vars_t2 = 0.0
    vars_t3 = 0.0

    for i, data in enumerate(trainloader, 0):
        # eff way
        vars_t1 += var(data[3].numpy(), t1_mean[0])
        vars_t2 += var(data[4].numpy(), t2_mean[0])
        vars_t3 += var(data[5].numpy(), t3_mean[0])

        
        if i % 100 == 99:
            print(f'done: [{i+1} / {len(trainloader)}]')
    
    
    logging.info('w2 std')
    t1_std = (vars_t1 / (i+1))**0.5
    t2_std = (vars_t2 / (i+1))**0.5
    t3_std = (vars_t3 / (i+1))**0.5

    logging.info(f'w2 means: {t1_mean}, {t2_mean}, {t3_mean}')
    logging.info(f'w2 stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')
    
    
def mean_stds_br3_ch1_mem_eff_w3(trainloader):

    logging.info('Calculating w3 mean')
    
    task_1 = None
    task_2 = None
    task_3 = None


    for i, data in enumerate(trainloader, 0):
        
        if task_1 is None:
            task_1 = np.sum(data[6].numpy(), axis=0) / data[6].shape[0]
        else:
            task_1 = np.add(task_1, np.sum(data[6].numpy(), axis=0) / data[6].shape[0])
        
        if task_2 is None:
            # task_2 = data[1].numpy()
            task_2 = np.sum(data[7].numpy(), axis=0) / data[7].shape[0]
        else:
            # task_2 = np.append(task_2, data[1].numpy(), axis=0)
            task_2 = np.add(task_2, np.sum(data[7].numpy(), axis=0) / data[7].shape[0])
        
        if task_3 is None:
            # task_3 = data[2].numpy()
            task_3 = np.sum(data[8].numpy(), axis=0) / data[8].shape[0]
        else:
            # task_3 = np.append(task_3, data[2].numpy(), axis=0)
            task_3 = np.add(task_3, np.sum(data[8].numpy(), axis=0) / data[8].shape[0])
        
        
        if i % 100 == 99:
            logging.info(f'done: [{i+1} / {len(trainloader)}]')


    task_1 = task_1 / (i+1)
    task_2 = task_2 / (i+1)
    task_3 = task_3 / (i+1)
    

    logging.info('======================================')
    logging.info('w3 mean')
    # task_1 = x[0].numpy()
    t1_mean = task_1.mean(axis=(1,2))
    # t1_std = task_1.std(axis=(1,2))
    # logging.info(f'{t1_mean}, {t1_std}')

    # task_2 = x[1].numpy()
    t2_mean = task_2.mean(axis=(1,2))
    # t2_std = task_2.std(axis=(1,2))
    # logging.info(f'{t2_mean}, {t2_std}')

    # task_3 = x[2].numpy()
    t3_mean = task_3.mean(axis=(1,2))
    # t3_std = task_3.std(axis=(1,2))
    # logging.info(f'{t3_mean}, {t3_std}')

    logging.info(f'means: {t1_mean}, {t2_mean}, {t3_mean}')
    # logging.info(f'stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')


    logging.info('======================================')
    logging.info('Calculating std')


    def var(a, mean):

        # N = len(a)
        N = 1
        for i in a.shape:
            N *= i

        d2 = (a - mean)**2
        var = d2.sum() / (N)

        return var


    vars_t1 = 0.0
    vars_t2 = 0.0
    vars_t3 = 0.0

    for i, data in enumerate(trainloader, 0):
        # eff way
        vars_t1 += var(data[6].numpy(), t1_mean[0])
        vars_t2 += var(data[7].numpy(), t2_mean[0])
        vars_t3 += var(data[8].numpy(), t3_mean[0])

        
        if i % 100 == 99:
            print(f'done: [{i+1} / {len(trainloader)}]')
    
    
    logging.info('w3 std')
    t1_std = (vars_t1 / (i+1))**0.5
    t2_std = (vars_t2 / (i+1))**0.5
    t3_std = (vars_t3 / (i+1))**0.5

    logging.info(f'w3 means: {t1_mean}, {t2_mean}, {t3_mean}')
    logging.info(f'w3 stds: {t1_std}, {t2_std}, {t3_std}')
    logging.info('======================================')
    

def get_mean_stds(
    yaml_path, 
    x_train_load_path, 
    y_train_load_path, 
    ):
    """
    """
    app_config = load_yaml_file(yaml_path)

    # x_train_load_path = 'debug/6kd_cnn6_100ep_dWstand_8_1/dataset/x_train.csv'
    # y_train_load_path = 'debug/6kd_cnn6_100ep_dWstand_8_1/dataset/y_train.csv'

    X_train = pd.read_csv(x_train_load_path)
    print(X_train.head())

    y_train = pd.read_csv(y_train_load_path)
    print(y_train.head())

    


    # set transform = None using config file normalization: '' setting.
    app_config['normalization'] = ''

    train_dataset = GenomicDataset(
        X_train, y_train, 
        app_config=app_config, 
    )


    batch_size = 20
    if app_config['batch_size']:
        batch_size = app_config['batch_size']

    num_workers = 2
    if app_config['train_dl_num_workers']:
        num_workers = app_config['train_dl_num_workers']
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    model_parallel_branches = cnn_model_metadata[app_config['cnn_model']]['parallel_branches']
    model_per_branch_channels = cnn_model_metadata[app_config['cnn_model']]['channel_per_branch']

    if model_parallel_branches == 3 and model_per_branch_channels == 1:
        mean_stds_br3_ch1(trainloader)
    
    elif model_parallel_branches == 9 and model_per_branch_channels == 1:
        mean_stds_br9_ch1(trainloader)

    elif model_parallel_branches == 3 and model_per_branch_channels == 3:
        # we need to figure out model_parallel_branches == 9 and model_per_branch_channels == 1 values
        app_config['cnn_model'] = 'Net_6'
        
        train_dataset = GenomicDataset(
            X_train, y_train, 
            app_config=app_config, 
        )

        batch_size = 20
        trainloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=2
        )
        mean_stds_br9_ch1(trainloader)
    
    else:
        logging.info('TODO')


def get_mean_stds_transfer_learning(
    yaml_path, 
    x_train_load_path, 
    y_train_load_path, 
    ):
    """
    """
    app_config = load_yaml_file(yaml_path)

    # x_train_load_path = 'debug/6kd_cnn6_100ep_dWstand_8_1/dataset/x_train.csv'
    # y_train_load_path = 'debug/6kd_cnn6_100ep_dWstand_8_1/dataset/y_train.csv'

    X_train = pd.read_csv(x_train_load_path)
    print(X_train.head())

    y_train = pd.read_csv(y_train_load_path)
    print(y_train.head())

    
    # set transform = None using config file normalization: '' setting.
    app_config['normalization'] = ''
    
    train_dataset = GenomicTLDataset(
        X_train, y_train, 
        app_config=app_config, 
    )


    batch_size = 20
    if app_config['batch_size']:
        batch_size = app_config['batch_size']

    num_workers = 2
    if app_config['train_dl_num_workers']:
        num_workers = app_config['train_dl_num_workers']
    
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )

    # mean_stds_br3_ch1(trainloader)
    mean_stds_br3_ch1_mem_eff(trainloader)
    mean_stds_br3_ch1_mem_eff_w2(trainloader)
    mean_stds_br3_ch1_mem_eff_w3(trainloader)


def get_mean_stds_main(yaml_path='configs/means_stds_calculation_debug.yaml'):
    try:

        app_config = load_yaml_file(yaml_path)
        train_dataset, val_dataset, test_dataset = generate_datasets(app_config, dataset_type='transfer_learning')
        # train_dataset, val_dataset, test_dataset = generate_datasets(app_config)

        dataset_ref_path = os.path.join(app_config['output_save_path'], 'dataset')

        x_train_load_path = os.path.join(dataset_ref_path, 'x_train.csv')
        y_train_load_path = os.path.join(dataset_ref_path, 'y_train.csv')

        # x_train_load_path = 'debug/ceu5500_meanstd_calc/dataset/x_train.csv'
        # y_train_load_path = 'debug/ceu5500_meanstd_calc/dataset/y_train.csv'
        
        # get_mean_stds(
        #     yaml_path, 
        #     x_train_load_path, 
        #     y_train_load_path, 
        # )

        get_mean_stds_transfer_learning(
            yaml_path, 
            x_train_load_path, 
            y_train_load_path, 
        )

        logging.info('--------------------------------------------------------------------------------')

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)


def main(app_config):
    try:
        train_dataset, val_dataset, test_dataset = generate_datasets(app_config)

        iter_demo = iter(train_dataset)
        # train_dataloader = DataLoader(train_dataset, batch_size=app_config['batch_size'], shuffle=True, num_workers=2)
        # iter_demo = iter(train_dataloader)

        for i in range(10):
            _ = next(iter_demo)

        logging.info('--------------------------------------------------------------------------------')

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)


def infer_main():
    try:
        app_config = load_yaml_file('workspace/configs/infer_chromosome2_debug.yaml')
        infer_dataset = generate_infer_datasets(app_config)

        iter_demo = iter(infer_dataset)

        for i in range(10):
            _ = next(iter_demo)

        logging.info('--------------------------------------------------------------------------------')

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)


######################################### util functions end #########################################

if __name__ == "__main__":
    # read config
    args = argument_parser()
    app_config = load_yaml_file(args.config)

    # create logs
    create_logfile(app_config['output_save_path'])


    logging.info('Starting application')
    logging.info('Reading config parameters from file {}'.format(args.config))
    logging.info(
        'Application configuration parameters: {}'.format(
            app_config
        )
    )

    # https://discuss.pytorch.org/t/reproducibility-dataloader-shuffle-true-using-seeds/173836
    torch.manual_seed(app_config['seed'])

    # main(app_config)
    # infer_main()

    get_mean_stds_main(yaml_path=args.config)

    logging.info('--------------------------------------------------------------------------------')

