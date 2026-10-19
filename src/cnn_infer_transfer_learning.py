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
import math

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
from src.cnn_models import get_model
from src.utils import *
from src.constants import *
from src.cnn_genomic_datasets import generate_infer_datasets

######################################### import end #########################################


class CNN_infer_TL_model():
    def __init__(self, app_config, infer_dataset):
        
        self.app_config = app_config
        self.batch_size = self.app_config['batch_size']

        self.init_folders()

        self.infer_report_dict = {
            'infered_class': [],
            'position': [],
            'chromosome': []
        }

        self.init_infer_data_loader(infer_dataset)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        logging.info('Found device: {}'.format(self.device))


        self.model = get_model(self.app_config)
        # self.model.show_summary()
        
        ### set to device before init optimizer to solve the issue with cpu, gpu save, load.
        ### https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2
        self.model.to(self.device)

        # criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()

        # create your optimizer
        # optimizer = optim.SGD(net.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        # self.init_optimizer()

        self.load_model()

        logging.info(f'model loaded in: {next(self.model.parameters()).device} for inference')

    
    def init_optimizer(self):

        try:
            if 'optimizer' in self.app_config:
                
                if 'parameters' in self.app_config['optimizer'] and 'name' in self.app_config['optimizer']:
                    
                    parameters = self.app_config['optimizer']['parameters']
                    name = self.app_config['optimizer']['name']

                    if name == 'adam':
                        if parameters:
                            self.optimizer = optim.Adam(self.model.parameters(), **parameters)
                        else:
                            self.optimizer = optim.Adam(self.model.parameters())
                        logging.info(f'Using Adam optimizer {self.optimizer}')
                    else:
                        logging.warning('Wrong optimizer name used in application config')
                        logging.info('Using default optimizer for training model.')
                        self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
                        logging.info(f'Using SGD optimizer {self.optimizer}')
                
                else:
                    logging.warning('No optimizer details used in application config')
                    logging.info('Using default optimizer for training model.')
                    self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
                    logging.info(f'Using SGD optimizer {self.optimizer}')

            else:
                logging.info('Using default optimizer for training model.')
                self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
                logging.info(f'Using SGD optimizer {self.optimizer}')
        
        except Exception as e:
            logging.error("An exception occurred while initializing optimizer")
            logging.error(e)
    

    def init_folders(self):
        """
        """
        self.model_path = self.app_config['output_save_path']
        os.makedirs(self.model_path, exist_ok = True)

        if "report_folder" in self.app_config and self.app_config['report_folder'] != '':
            report_folder = str(self.app_config['report_folder'])
        else:
            report_folder = 'infer_report'
        
        self.report_path = os.path.join(self.model_path, report_folder)
        os.makedirs(self.report_path, exist_ok=True)
        logging.info('Saving inference report to {}'.format(self.report_path))


    def init_infer_data_loader(self, infer_dataset):
        """
        """
        logging.info('Creating dataloader from dataset')
        infer_dl_num_workers = 2
        if "infer_dl_num_workers" in self.app_config and self.app_config['infer_dl_num_workers'] != '':
            infer_dl_num_workers = int(self.app_config['infer_dl_num_workers'])
        
        self.infer_dataloader = DataLoader(infer_dataset, batch_size=self.batch_size, shuffle=True, num_workers=infer_dl_num_workers)
        
        # original labels
        self.class_labels = infer_dataset.labels # not same as lb.classes_
        self.lb = infer_dataset.lb
        self.dataset_name = infer_dataset.dataset_name

        # self.model_parallel_channels = train_dataset.model_parallel_channels
        self.model_parallel_branches = infer_dataset.model_parallel_branches
        self.model_per_branch_channels = infer_dataset.model_per_branch_channels
        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels
    

    def load_model(self):
        """
        """
        if "load_pretrained_model" in self.app_config and self.app_config['load_pretrained_model'] != '':
            model_save_path = self.app_config['load_pretrained_model']
        else:
            
            folder_path = os.path.join(self.model_path, 'model')
            model_save_path = os.path.join(folder_path, 'model.pth')

        
        if os.path.isfile(model_save_path):

            # Load model checkpoint
            checkpoint = torch.load(model_save_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            # self.start_epoch = checkpoint['epoch'] + 1
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


            logging.info('Loaded saved model: {}'.format(model_save_path))

        else:
            logging.warning('Saved model path does not exist! {}'.format(model_save_path))
            logging.info('Starting model training from the beginning')


    def model_output(self, data):
        """
        """
        if self.model_parallel_branches == 9 and self.model_per_branch_channels == 3:
            t1_images, t2_images, t3_images, \
                w2_t1_images, w2_t2_images, w2_t3_images, \
                w3_t1_images, w3_t2_images, w3_t3_images, position, chromosome_label = data
            
            # set data to right device
            t1_images = t1_images.to(self.device)
            t2_images = t2_images.to(self.device)
            t3_images = t3_images.to(self.device)
            
            w2_t1_images = w2_t1_images.to(self.device)
            w2_t2_images = w2_t2_images.to(self.device)
            w2_t3_images = w2_t3_images.to(self.device)
            
            w3_t1_images = w3_t1_images.to(self.device)
            w3_t2_images = w3_t2_images.to(self.device)
            w3_t3_images = w3_t3_images.to(self.device)

            position = position.squeeze()
            
            # calculate outputs by running images through the network
            outputs = self.model(
                t1_images, t2_images, t3_images, 
                w2_t1_images, w2_t2_images, w2_t3_images, 
                w3_t1_images, w3_t2_images, w3_t3_images, 
                )
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            t1_images, t2_images, t3_images, position, chromosome_label = data
            
            # set data to right device
            t1_images = t1_images.to(self.device)
            t2_images = t2_images.to(self.device)
            t3_images = t3_images.to(self.device)
            
            position = position.squeeze()
            
            # calculate outputs by running images through the network
            outputs = self.model(t1_images, t2_images, t3_images)
        
        else:
            logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                self.model_parallel_branches, self.model_per_branch_channels))
        

        return outputs, position, chromosome_label


    def infer(self):
        """
        """
        logging.info('--------- start infering ---------')
        self.model.eval()
        infer_data_len = len(self.infer_dataloader)
        
        predicted_classes = []
        prediction_scores = []
        positions = []
        chromosome_labels = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(self.infer_dataloader, 0):
                
                # calculate outputs by running images through the network
                outputs, position, chromosome_label = self.model_output(data)
                
                
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                # predictions

                s = nn.Softmax(dim = 1)
                conf_score_preds = s(outputs)


                predicted_classes.extend(predicted.tolist())

                prediction_scores.extend(conf_score_preds.tolist())

                positions.extend(position.tolist())
                chromosome_labels.extend(chromosome_label)

                if i % 50 == 49:    # print every 2000 mini-batches
                    logging.info(f'\t[{i + 1:5d}/{infer_data_len}] inferred')
        
        return predicted_classes, prediction_scores, chromosome_labels, positions


    def infer_report(self, predicted_classes, prediction_scores, chromosome_labels, positions):
        """
        """
        # save the prediction scores
        if True:
            df = pd.DataFrame(
                prediction_scores, 
                columns = self.lb.classes_
            )

            df['predicted_class_id'] = predicted_classes

            predicted_class = []
            for o in predicted_classes:
                predicted_class.append(self.lb.classes_[o])
            df['predicted_class'] = predicted_class

            df['position'] = positions
            df['chromosome'] = chromosome_labels

            df = df.sort_values(['chromosome', 'position'])

            # report_path = os.path.join(self.model_path, 'infer_report')
            # os.makedirs(report_path, exist_ok=True)
            # logging.info('saving test report to {}'.format(report_path))
            file_name = os.path.join(self.report_path, 'infer_prediction_scores.csv')
            df.to_csv(file_name, index=False)
            logging.info('saving test prediction with confidence scores here {}'.format(file_name))
        
        
        # plot pred probability of all class
        if True:

            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
            colors_4 = ['r', 'b', 'g', 'y']

            for chromosome in self.app_config['chromosomes']:

                ch_df = df[df['chromosome'] == chromosome]
                
                plt.figure(figsize=(10,6))
                for tp_class, color in zip(tp_classes, colors_4):
                    pred_proba = ch_df[tp_class]
                    chrom_position = ch_df['position']

                    
                    # plt.scatter(chrom_position, pred_prob_sweep, color=color, label='sweeps')
                    plt.plot(chrom_position, pred_proba, color=color, label=tp_class)
                    plt.yticks(np.arange(0, 1.1, 0.1))
                    plt.title(chromosome + ' Prediction Probability')
                    plt.xlabel('Chromosome Position')
                    plt.ylabel('Prediction Probability of all class')
                    plt.legend()
                    plt.grid()

                file_name = chromosome + '_prediction_probability_all.pdf'
                plot_full_path = os.path.join(self.report_path, file_name)
                plt.savefig(plot_full_path)
                plt.close()
                logging.info('Prediction Probability of all plot saved here: {}'.format(plot_full_path))

        # plot pred probability of all class separately
        if True:
            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
            colors_4 = ['r', 'b', 'g', 'y']

            for chromosome in self.app_config['chromosomes']:
                ch_df = df[df['chromosome'] == chromosome]

                for tp_class, color in zip(tp_classes, colors_4):
                    pred_proba = ch_df[tp_class]
                    chrom_position = ch_df['position']

                    plt.figure(figsize=(10,6))
                    # plt.scatter(chrom_position, pred_prob_sweep, color=color, label='sweeps')
                    plt.plot(chrom_position, pred_proba, color=color, label=tp_class)
                    plt.yticks(np.arange(0, 1.1, 0.1))
                    plt.title(chromosome + ' Prediction Probability')
                    plt.xlabel('Chromosome Position')
                    plt.ylabel(f'Prediction Probability of {tp_class} class')
                    plt.legend()
                    plt.grid()

                    file_name = f'{chromosome}_prediction_probability_{tp_class}.pdf'
                    plot_full_path = os.path.join(self.report_path, file_name)
                    plt.savefig(plot_full_path)
                    plt.close()
                    logging.info('Prediction Probability of {} plot saved here: {}'.format(tp_class, plot_full_path))


        # scatter plot pred probability separately
        if True:
            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
            colors_4 = ['r', 'b', 'g', 'y']

            for chromosome in self.app_config['chromosomes']:
                ch_df = df[df['chromosome'] == chromosome]

                for tp_class, color in zip(tp_classes, colors_4):
                    pred_proba = ch_df[tp_class]
                    chrom_position = ch_df['position']
                    s = [0.25 for i in range(len(pred_proba))]

                    plt.figure(figsize=(10,6))
                    # plt.scatter(chrom_position, pred_prob_sweep, color=color, label='sweeps')
                    plt.scatter(chrom_position, pred_proba, color=color, label=tp_class, s=s)
                    # plt.plot(chrom_position, pred_proba, color=color, label=tp_class)
                    plt.yticks(np.arange(0, 1.1, 0.1))
                    plt.title(chromosome + ' Prediction Probability')
                    plt.xlabel('Chromosome Position')
                    plt.ylabel(f'Prediction Probability of {tp_class} class')
                    plt.legend()
                    plt.grid()
                    # plt.plot()
                    # break

                    file_name = f'{chromosome}_scatter_prediction_probability_{tp_class}.pdf'
                    plot_full_path = os.path.join(self.report_path, file_name)
                    # plt.savefig(plot_full_path, dpi=300)
                    plt.savefig(plot_full_path)
                    plt.close()


        # log odds 
        def log_proba(pred_proba):
            return np.log10(pred_proba / (1 - pred_proba + 1e-10))
        
        # find max log odds in data
        def find_max_log_odds(df, tp_classes, chromosomes=['chromosome_2', 'chromosome_6']):
            max_val = -100
            # tp_classes = ['AI', 'BalSel', 'Sweeps']

            for chromosome in chromosomes:
            # for chromosome in ['chromosome_2']:
                ch_df = df[df['chromosome'] == chromosome]

                for tp_class in tp_classes:
                    pred_proba = ch_df[tp_class]
                    log_pred_proba = log_proba(pred_proba)
                    max_log_pred_proba = max(log_pred_proba)

                    if max_log_pred_proba > max_val:
                        max_val = max_log_pred_proba
            
            return math.ceil(max_val)


        # plot log odds pred probability of all class separately
        if True:
            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps'], self.dataset_name)
            colors_4 = ['r', 'b', 'g']

            # find max log odds
            max_y = find_max_log_odds(df, tp_classes=tp_classes, chromosomes=self.app_config['chromosomes'])

            for chromosome in self.app_config['chromosomes']:
                ch_df = df[df['chromosome'] == chromosome]

                for tp_class, color in zip(tp_classes, colors_4):
                    pred_proba = ch_df[tp_class]
                    log_pred_proba = log_proba(pred_proba)
                    chrom_position = ch_df['position']

                    flag = log_pred_proba > 0
                    log_pred_proba_trunc = log_pred_proba[flag]
                    chrom_position_trunc = chrom_position[flag]
                    s = [0.5 for i in range(len(log_pred_proba_trunc))]

                    plt.figure(figsize=(10,6))
                    plt.scatter(chrom_position_trunc, log_pred_proba_trunc, color=color, label=tp_class, s=s)
                    plt.yticks(np.arange(0, max_y + 1, 1))
                    # plt.ylim(6)
                    
                    plt.title(chromosome + ' Prediction log Probability')
                    plt.xlabel('Chromosome Position')
                    l = f'$\log_{{10}} (p({tp_class}) / (1 - p({tp_class})))$ of {tp_class} class'
                    plt.ylabel(r"{}".format(l))
                    plt.ylim(0, max_y)
                    
                    plt.legend()
                    plt.grid()
                    # plt.plot()
                    
                    file_name = f'{chromosome}_prediction_log_probability_sscale_{tp_class}.pdf'
                    plot_full_path = os.path.join(self.report_path, file_name)
                    # plt.savefig(plot_full_path, dpi=300)
                    plt.savefig(plot_full_path)
                    plt.close()


        if True:
            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
            colors_4 = ['r', 'b', 'g', 'y']

            for chromosome in self.app_config['chromosomes']:
                ch_df = df[df['chromosome'] == chromosome]

                for tp_class, color in zip(tp_classes, colors_4):
                    pred_proba = ch_df[tp_class]

                    # plt.figure(figsize=(10,6))
                    counts, edges, bars = plt.hist(pred_proba, bins=np.arange(0,1.1,.1))
                    plt.bar_label(bars)
                    plt.xticks(np.arange(0, 1.1, 0.1))
                    plt.yticks(np.arange(0, 15000, 2000))
                    plt.title(chromosome + ' Prediction Probability Distribution')
                    plt.xlabel(f'Prediction Probability of {tp_class}')
                    plt.ylabel('num of observation')
                    # plt.legend()
                    plt.grid()

                    file_name = f'{chromosome}_prediction_probability_distribution_{tp_class}.pdf'
                    plot_full_path = os.path.join(self.report_path, file_name)
                    plt.savefig(plot_full_path)
                    plt.close()
                    logging.info('Prediction Probability distribution of {} plot saved here: {}'.format(tp_class, plot_full_path))
        
        # gene inference prediction plot
        if True:
            
            report_path = self.report_path
            tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
            colors_4 = ['r', 'b', 'g', 'y']

            for chromosome, zones in gene_mapping.items():
                # print(chromosome, zones)
                for zone_name, zone_data in zones.items():
                    # print(zone_name, zone_data)
                    plot_chromosome_zone(df, chromosome, self.app_config, zone_name, zone_data, tp_classes, colors_4, report_path)


def main(app_config):
    try:
        infer_dataset = generate_infer_datasets(app_config, dataset_type='transfer_learning')

        cnn = CNN_infer_TL_model(app_config, infer_dataset)

        predicted_classes, prediction_scores, chromosome_labels, positions = cnn.infer()
        cnn.infer_report(predicted_classes, prediction_scores, chromosome_labels, positions)

        logging.info('--------------------------------------------------------------------------------')

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)
        logging.error(repr(traceback.extract_tb(e.__traceback__)))


#######################################
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

    main(app_config)

    logging.info('--------------------------------------------------------------------------------')

