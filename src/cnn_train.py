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
from src.cnn_models import get_model
from src.utils import load_yaml_file, argument_parser, create_logfile, original_labels_to_dataset_labels
from src.cnn_genomic_datasets import generate_datasets

######################################### import end #########################################


class CNN_Model():
    def __init__(
            self, 
            app_config,
            train_dataset, 
            val_dataset, 
            test_dataset
        ):
        """
        """
        self.app_config = app_config
        self.batch_size = self.app_config['batch_size']
        self.epochs = self.app_config['epochs']
        self.early_stop_threshold = self.app_config['early_stop_threshold']
        self.model_path = self.app_config['output_save_path']
        os.makedirs(self.model_path, exist_ok = True)

        self.training_report = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        self.init_data_loader(train_dataset, val_dataset, test_dataset)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # self.device = 'cpu'
        # Assuming that we are on a CUDA machine, this should print a CUDA device:
        logging.info('Running on device: {}'.format(self.device))


        self.model = get_model(self.app_config)
        
        self.model.show_summary()
        
        ### set to device before init optimizer to solve the issue with cpu, gpu save, load.
        ### https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2
        self.model.to(self.device)

        # criterion = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss()

        # # create your optimizer
        # # optimizer = optim.SGD(net.parameters(), lr=0.01)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
        self.init_optimizer()

        self.start_epoch = 0
        if self.app_config['resume_training']:
            self.load_model()
        

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


    def init_data_loader(self, train_dataset, val_dataset, test_dataset):
        """
        """
        logging.info('Creating dataloader from dataset')
        # check num_workers for each dataloader
        train_dl_num_workers = 2
        if "train_dl_num_workers" in self.app_config and self.app_config['train_dl_num_workers'] != '':
            train_dl_num_workers = int(self.app_config['train_dl_num_workers'])
        
        val_dl_num_workers = 2
        if "val_dl_num_workers" in self.app_config and self.app_config['val_dl_num_workers'] != '':
            val_dl_num_workers = int(self.app_config['val_dl_num_workers'])
        
        test_dl_num_workers = 2
        if "test_dl_num_workers" in self.app_config and self.app_config['test_dl_num_workers'] != '':
            test_dl_num_workers = int(self.app_config['test_dl_num_workers'])

        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=train_dl_num_workers)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=val_dl_num_workers)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, num_workers=test_dl_num_workers)

        # original labels
        self.class_labels = train_dataset.labels # not same as lb.classes_
        self.lb = train_dataset.lb
        self.dataset_name = train_dataset.dataset_name

        # self.model_parallel_channels = train_dataset.model_parallel_channels
        self.model_parallel_branches = train_dataset.model_parallel_branches
        self.model_per_branch_channels = train_dataset.model_per_branch_channels
        self.model_total_tasks_files = self.model_parallel_branches * self.model_per_branch_channels


    def model_output(self, data):
        """
        """
        if self.model_parallel_branches == 3 and self.model_per_branch_channels == 1:
            t1_images, t2_images, t3_images, target = data
            
            # set data to right device
            t1_images = t1_images.to(self.device)
            t2_images = t2_images.to(self.device)
            t3_images = t3_images.to(self.device)
            target = target.to(self.device)
            target = target.squeeze()
            
            # calculate outputs by running images through the network
            outputs = self.model(t1_images, t2_images, t3_images)
        
        elif self.model_parallel_branches == 9 and self.model_per_branch_channels == 1:
            t1_images, t2_images, t3_images, \
                w2_t1_images, w2_t2_images, w2_t3_images, \
                w3_t1_images, w3_t2_images, w3_t3_images, target = data
            
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
            target = target.to(self.device)
            target = target.squeeze()
            
            # calculate outputs by running images through the network
            outputs = self.model(
                t1_images, t2_images, t3_images, 
                w2_t1_images, w2_t2_images, w2_t3_images, 
                w3_t1_images, w3_t2_images, w3_t3_images, 
                )
        
        elif self.model_parallel_branches == 3 and self.model_per_branch_channels == 3:
            t1_images, t2_images, t3_images, target = data
            
            # set data to right device
            t1_images = t1_images.to(self.device)
            t2_images = t2_images.to(self.device)
            t3_images = t3_images.to(self.device)
            target = target.to(self.device)
            target = target.squeeze()
            
            # calculate outputs by running images through the network
            outputs = self.model(t1_images, t2_images, t3_images)
        
        else:
            logging.error('transformation definition not implemented for branch {} and channel {}'.format(
                self.model_parallel_branches, self.model_per_branch_channels))

        return outputs, target


    def train(self):
        """
        """

        # set model to right device
        self.model.to(self.device)

        logging.info('--------- start training ---------')
        training_data_len = len(self.train_dataloader)

        if self.start_epoch == self.epochs:
            logging.info('Model is already trained to desired epoch number')
            return 0

        for epoch in range(self.start_epoch, self.epochs):  # loop over the dataset multiple times
            
            # train
            running_loss = 0.0
            for i, data in enumerate(self.train_dataloader, 0):

                self.optimizer.zero_grad()   # zero the gradient buffers

                outputs, target = self.model_output(data)

                loss = self.criterion(outputs.float(), target.float())
                # print('loss: ', loss)

                loss.backward()
                self.optimizer.step()    # Does the update

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:    # print every 2000 mini-batches
                    logging.info(f'\t[{epoch + 1:3d}, {i + 1:5d}/{training_data_len}] loss: {running_loss / (i + 1):.3f}')
                    
            running_loss = running_loss / (i + 1)
            

            if self.app_config['validation']:
                # val
                val_loss, val_accuracy = self.validation()

                logging.info(f'[{epoch + 1:3d}] train loss: {running_loss:.3f} val loss: {val_loss:.3f}')
                
                self.training_report['train_loss'].append(running_loss)
                self.training_report['val_loss'].append(val_loss)
                self.training_report['val_acc'].append(val_accuracy)
            
            else:
                
                logging.info(f'[{epoch + 1:3d}] train loss: {running_loss:.3f}')
                
                self.training_report['train_loss'].append(running_loss)
                self.training_report['val_loss'].append(-1)
                self.training_report['val_acc'].append(-1)
            
            # periodically save model
            # self.save_model_periodically(epoch, loss, 2)
            self.save_model_periodically(epoch, loss)


        logging.info('Finished Training')
        
        ##save training report
        self.save_model(epoch, loss)
        self.save_training_report()

        return 0


    def validation(self):
        """
        """

        correct = 0
        total = 0
        running_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(self.val_dataloader, 0):
                
                outputs, target = self.model_output(data)
                
                # loss 
                loss = self.criterion(outputs.float(), target.float())
                running_loss += loss.item()
                
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                _, labels = torch.max(target, 1)
                correct += (predicted == labels).sum().item()

        return running_loss / (i + 1), 100 * correct // total


    def test(self):
        """
        """
        logging.info('--------- start testing ---------')
        test_data_len = len(self.test_dataloader)
        correct = 0
        total = 0
        running_loss = 0.0
        predicted_classes = []
        prediction_scores = []
        original_classes = []
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader, 0):
                
                outputs, target = self.model_output(data)
                
                # loss 
                loss = self.criterion(outputs.float(), target.float())
                running_loss += loss.item()
                
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                # predictions

                s = nn.Softmax(dim = 1)
                conf_score_preds = s(outputs)


                total += target.size(0)
                _, labels = torch.max(target, 1)
                correct += (predicted == labels).sum().item()

                predicted_classes.extend(predicted.tolist())
                original_classes.extend(labels.tolist())
                prediction_scores.extend(conf_score_preds.tolist())

                if i % 50 == 49:    # print every 2000 mini-batches
                    logging.info(f'\t[{i + 1:5d}/{test_data_len}] loss: {running_loss / (i + 1):.3f}')

        total_loss = running_loss / (i + 1)
        accuracy = 100 * correct // total
        
        logging.info(f'Total loss: {total_loss:.3f} Accuracy: {accuracy}')
        
        return original_classes, predicted_classes, total_loss, accuracy, prediction_scores


    def save_model(self, epoch, loss, file_name=None):
        """
        https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
        https://pytorch.org/tutorials/beginner/saving_loading_models.html#:~:text=to%20the%20dictionary.-,A%20common%20PyTorch%20convention%20is%20to%20save%20these%20checkpoints%20using,dictionary%20as%20you%20would%20expect.
        """

        if file_name is None:
            file_name = 'model.pth'

        folder_path = os.path.join(self.model_path, 'model')
        os.makedirs(folder_path, exist_ok = True)

        model_save_path = os.path.join(folder_path, file_name)
        
        torch.save(
            {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            }, 
            model_save_path
        )
        logging.info('saved model at: {}'.format(model_save_path))


    def save_training_report(self):
        """
        """
        report_path = os.path.join(self.model_path, 'training_report')
        os.makedirs(report_path, exist_ok=True)
        logging.info('saving training report to {}'.format(report_path))

        ### dump training report dict as json
        with open(os.path.join(report_path, 'training_report.json'), 'w') as fp:
            json.dump(self.training_report, fp)


        if self.app_config['validation']:

            
            # find lowest loss epoch
            index_min = np.argmin(self.training_report['val_loss'])
            logging.info(f'Lowest validation loss epoch: {index_min + 1}')


            ### loss curve
            x = [i + 1 for i in range(len(self.training_report['train_loss']))]
            plt.plot(x, self.training_report['train_loss'], label = "train loss")
            plt.plot(x, self.training_report['val_loss'], label = "val loss")
            step = max(1, int(len(self.training_report['train_loss']) / 10))
            plt.xticks(np.arange(min(x), max(x)+1, step))
            plt.plot()
            plt.title('Loss curve')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.grid()
            # plt.show()
            # file_name = 'loss_curve.png'
            file_name = 'loss_curve.pdf'
            plot_full_path = os.path.join(report_path, file_name)
            # will save in this directory
            # plt.savefig(plot_full_path)
            plt.savefig(plot_full_path, format="pdf")
            plt.close()

            ### validation accuracy
            x = [i + 1 for i in range(len(self.training_report['val_acc']))]
            plt.plot(x, self.training_report['val_acc'], label = "val accuracy")
            step = max(1, int(len(self.training_report['val_acc']) / 10))
            plt.xticks(np.arange(min(x), max(x)+1, step))
            plt.title('validation accuracy curve')
            plt.xlabel('epochs')
            plt.ylabel('accuracy')
            plt.legend()
            plt.grid()
            # plt.show()
            # file_name = 'val_accuracy_curve.png'
            file_name = 'val_accuracy_curve.pdf'
            plot_full_path = os.path.join(report_path, file_name)
            # will save in this directory
            plt.savefig(plot_full_path, format="pdf")
            # plt.savefig(plot_full_path)
            plt.close()

        else:
            ### loss curve
            x = [i + 1 for i in range(len(self.training_report['train_loss']))]
            plt.plot(x, self.training_report['train_loss'], label = "train loss")
            # plt.plot(x, self.training_report['val_loss'], label = "val loss")
            step = max(1, int(len(self.training_report['train_loss']) / 10))
            plt.xticks(np.arange(min(x), max(x)+1, step))
            plt.plot()
            plt.title('Loss curve')
            plt.xlabel('epochs')
            plt.ylabel('loss')
            plt.legend()
            plt.grid()
            # plt.show()
            # file_name = 'loss_curve.png'
            file_name = 'loss_curve.pdf'
            plot_full_path = os.path.join(report_path, file_name)
            # will save in this directory
            # plt.savefig(plot_full_path)
            plt.savefig(plot_full_path, format="pdf")
            plt.close()


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
            self.start_epoch = checkpoint['epoch'] + 1
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


            ## if only test
            if "only_test" in self.app_config and self.app_config['only_test']:
                logging.info('Loaded saved model: {}'.format(model_save_path))
                logging.info('Resuming training from epoch {}'.format(self.start_epoch + 1))
                # logging.info('Training report history loaded from : {}'.format(file_path))
                logging.info('Only loading model for test purposes so no training report loading is necessary')
                self.training_report = {}
                logging.info('Loaded Training report: {}'.format(self.training_report))
                return

            
            report_path = os.path.join(self.model_path, 'training_report')
            file_path = os.path.join(report_path, 'training_report.json')
            # Opening JSON file
            with open(file_path) as json_file:
                self.training_report = json.load(json_file)


            logging.info('Loaded saved model: {}'.format(model_save_path))
            logging.info('Resuming training from epoch {}'.format(self.start_epoch + 1))
            logging.info('Training report history loaded from : {}'.format(file_path))
            logging.info('Loaded Training report: {}'.format(self.training_report))

        else:
            logging.warning('Saved model path does not exist! {}'.format(model_save_path))
            logging.info('Starting model training from the beginning')


    def save_model_periodically(self, epoch, loss, period=10):
        """
        """
        # save every 10 epoch
        if epoch % period == (period - 1):
            logging.info('Saving model at epoch: {}'.format(epoch + 1))
            file_name = f'model_epoch_{str(epoch + 1)}.pth'
            self.save_model(epoch, loss, file_name=file_name)
            self.save_training_report()
        

    def test_report(self, original_classes, predicted_classes, prediction_scores):
        """
        """
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.metrics import ConfusionMatrixDisplay

        if True:
            # save original classes and predicted classes 
            test_report = {
                'original_classes': original_classes,
                'predicted_classes': predicted_classes
            }
            report_path = os.path.join(self.model_path, 'training_report')
            os.makedirs(report_path, exist_ok=True)
            logging.info('saving test report to {}'.format(report_path))

            ### dump training report dict as json
            with open(os.path.join(report_path, 'test_report.json'), 'w') as fp:
                json.dump(test_report, fp)
        
        
        # save the prediction scores
        if True:
            df = pd.DataFrame(
                prediction_scores, 
                columns = self.lb.classes_
            )
            original_class = []
            for o in original_classes:
                original_class.append(self.lb.classes_[o])

            df['original_class_id'] = original_classes
            df['original_class'] = original_class
            df['predicted_class_id'] = predicted_classes

            report_path = os.path.join(self.model_path, 'training_report')
            os.makedirs(report_path, exist_ok=True)
            # logging.info('saving test report to {}'.format(report_path))
            file_name = os.path.join(report_path, 'test_prediction_scores.csv')
            df.to_csv(file_name, index=False)
            logging.info('saving test prediction with confidence scores here {}'.format(file_name))


            # plot roc curve
            # one vs Neutrality 
            if 'BgSel' not in self.dataset_name:
                tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps'], self.dataset_name)
                fp_class = 'Neut'
                # fp_class = 'BgSel'

                colors_3 = ['r', 'b', 'g']

                for tp_class, color in zip(tp_classes, colors_3):
                    tp_class_df = df.loc[(df['original_class'] == tp_class)]
                    tp_cls_probs = np.array(tp_class_df[tp_class])

                    fp_class_df = df.loc[(df['original_class'] == fp_class)]
                    fp_cls_probs = np.array(fp_class_df[tp_class])

                    fpr = np.linspace(0, 1, num=1001, endpoint=True)
                    tpr = [0]
                    for i in range(1, len(fpr) - 1):
                        tpr_tp_cls = np.average(tp_cls_probs >= np.quantile(fp_cls_probs, 1 - fpr[i]))
                        tpr.append(tpr_tp_cls)
                    tpr.append(1)

                    # random_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

                    # plt.plot(fpr, tpr, color=random_color, label=tp_class)
                    plt.plot(fpr, tpr, color=color, label=tp_class)
                    plt.title('ROC curve (one vs neutrailty)')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    plt.grid()
                    # plt.show()

                # plt.show()
                # file_name = 'roc_selection_vs_neutrality.png'
                file_name = 'roc_selection_vs_neutrality.pdf'
                plot_full_path = os.path.join(report_path, file_name)
                # plt.savefig(plot_full_path)
                plt.savefig(plot_full_path, format="pdf")
                plt.close()
                logging.info('selection vs neutrality roc curve saved here: {}'.format(plot_full_path))


            # one vs rest 
            if 'BgSel' not in self.dataset_name:
            # if False:
                tp_classes = original_labels_to_dataset_labels(['AI', 'BalSel', 'Sweeps', 'Neut'], self.dataset_name)
                fp_class = 'Neut'

                colors_4 = ['r', 'b', 'g', 'y']

                for tp_class, color in zip(tp_classes, colors_4):
                    
                    tp_class_df = df.loc[(df['original_class'] == tp_class)]
                    tp_cls_probs = np.array(tp_class_df[tp_class])

                    fp_class_df = df.loc[(df['original_class'] != tp_class)]
                    fp_cls_probs = np.array(fp_class_df[tp_class])

                    fpr = np.linspace(0, 1, num=1001, endpoint=True)
                    tpr = [0]

                    for i in range(1, len(fpr) - 1):
                        tpr_tp_cls = np.average(tp_cls_probs >= np.quantile(fp_cls_probs, 1 - fpr[i]))
                        tpr.append(tpr_tp_cls)
                    tpr.append(1)

                    # random_color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))

                    # plt.plot(fpr, tpr, color=random_color, label=tp_class)
                    plt.plot(fpr, tpr, color=color, label=tp_class)
                    plt.title('ROC curve (one vs rest)')
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.legend()
                    # plt.grid() # TODO # verify the fix

                    # plt.show()

                # plt.show()
                plt.grid()
                # file_name = 'roc_selection_vs_all.png'
                file_name = 'roc_selection_vs_all.pdf'
                plot_full_path = os.path.join(report_path, file_name)
                # plt.savefig(plot_full_path)
                plt.savefig(plot_full_path, format="pdf")
                plt.close()
                logging.info('selection vs all roc curve saved here: {}'.format(plot_full_path))


        cr = classification_report(original_classes, predicted_classes)
        cm = confusion_matrix(original_classes, predicted_classes, normalize='true') * 100

        logging.info('Classification report: \n{}'.format(cr))
        logging.info('Confusion matrix: \n{}'.format(cm))


        # plot confusion matrix
        if True:
            report_path = os.path.join(self.model_path, 'training_report')
            os.makedirs(report_path, exist_ok=True)
            
            df = pd.DataFrame(cm, columns=self.lb.classes_, index=self.lb.classes_)
            exp_labels = original_labels_to_dataset_labels(['Neut', 'BalSel', 'AI', 'Sweeps'], self.dataset_name)
            if 'BgSel' in self.dataset_name:
                exp_labels = ['BgSel', 'BalSel', 'AI', 'Sweeps']
            df = df.reindex(index=exp_labels)
            df = df.reindex(columns=exp_labels)
            
            cmap = sns.color_palette("Blues", n_colors=10)
            # cmap = sns.color_palette("crest", n_colors=10)
            sns.heatmap(
                df, 
                cmap=cmap, 
                annot=True, 
                vmin=0, vmax=100, 
                # xticklabels=labels, yticklabels=labels,
                fmt=".2f"
            )

            plt.xlabel('Predicted label')
            plt.ylabel('True label')

            # file_name = 'confusion_matrix.png'
            file_name = 'confusion_matrix.pdf'
            plot_full_path = os.path.join(report_path, file_name)
            # plt.savefig(plot_full_path)
            plt.savefig(plot_full_path, format="pdf")
            plt.close()
            logging.info('Confusion matrix saved here: {}'.format(plot_full_path))


        return cr, cm


def main(app_config):
    try:
        train_dataset, val_dataset, test_dataset = generate_datasets(app_config)

        cnn = CNN_Model(app_config, train_dataset, val_dataset, test_dataset)

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
        # traceback.print_exc(limit=2, file=sys.stdout)
        logging.error(repr(traceback.extract_tb(e.__traceback__)))
        # logging.error(repr(traceback.format_exception(e)))


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

