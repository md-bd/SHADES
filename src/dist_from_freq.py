"""
Author: Mohammad Khan
mkhan2024@fau.edu
"""
######################################### import start #########################################

import os
import sys
import time
import yaml
from argparse import ArgumentParser
import logging
from datetime import datetime
from pytz import timezone

# # ----- logger configuration
logging.getLogger().setLevel(logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


# https://stackoverflow.com/a/74656206
os.environ['NUMEXPR_MAX_THREADS'] = '8'

import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from numba import jit

sys.path.append('.')

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


def create_logfile(save_dir='.', timezone_str='EST'):
    """
    configure logger
    """
    # ----- logger configuration
    # logging.getLogger().setLevel(logging.INFO)
    # logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

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


def check_read(file_path):
    """check if the file meets our requirement of reading or not
    freq_CEU_sweep_1.csv -> Not read
    all_freq_avg_Sweeps.csv -> Read
    """
    file_name = os.path.basename(file_path)
    return True
    if 'all_freq_avg_' in file_name:
        return True
    else:
        return False

@jit
def hellinger_dot(p, q):
    """Hellinger distance between two discrete distributions. 
       Using numpy.
       For Python >= 3.5 only
       ref https://nbviewer.org/gist/Teagum/460a508cda99f9874e4ff828e1896862
    """
    z = np.sqrt(p) - np.sqrt(q)
    return np.sqrt(z @ z / 2)


def calc_dist(current_window, second_window, method='Hellinger'):
    """
    """
    dist = None
    if method == 'Hellinger':
        dist = hellinger_dot(current_window, second_window)
    return dist


def calc_dist_each_col(data):
    """
    """
    total_windows = data.shape[1]
    total_rows = data.shape[0]
    freq_data = []
    
    max_len = -1
    for window in data.columns:

        window_freq = np.array(data.value_counts(window)) / total_rows
        freq_data.append(window_freq)

        if window_freq.shape[0] > max_len:
            max_len = window_freq.shape[0]
    
    # make all window_freq length same
    # not all window will have same diversity. 
    # some will contain less/more variation
    # add 0 to list with less than max_len length till it has max_len length
    for i in range(len(freq_data)):
        diff = max_len - freq_data[i].shape[0]

        if diff:
            freq_data[i] = np.append(freq_data[i], np.array([0] * diff))
    

    dist_data = []
    for i in range(total_windows):
        # logging.info('Current window: {}'.format(i))
        current_window = freq_data[i]

        window_dist = []
        # distance with other windows:
        for j in range(total_windows):
            second_window = freq_data[j]
            # calculate distance
            window_dist_i_j = calc_dist(current_window, second_window)
            window_dist.append(window_dist_i_j)
        
        dist_data.append(window_dist)
    
    # create dataframe and return dataframe
    col_names = ['d_w_' + str(i + 1) for i in range(total_windows)]
    df = pd.DataFrame(dict(zip(col_names, dist_data)))

    return df


def calc_dist_no_duplicate_col(data):
    """
    """
    total_windows = data.shape[1]
    total_rows = data.shape[0]
    freq_data = []
    
    max_len = -1
    for window in data.columns:

        window_freq = np.array(data.value_counts(window)) / total_rows
        freq_data.append(window_freq)

        if window_freq.shape[0] > max_len:
            max_len = window_freq.shape[0]
    
    # make all window_freq length same
    # not all window will have same diversity. 
    # some will contain less/more variation
    # add 0 to list with less than max_len length till it has max_len length
    for i in range(len(freq_data)):
        diff = max_len - freq_data[i].shape[0]

        if diff:
            freq_data[i] = np.append(freq_data[i], np.array([0] * diff))
    

    dist_data = np.zeros(shape=(total_windows, total_windows))
    for i in range(total_windows - 1):
        # logging.info('Current window: {}'.format(i))
        current_window = freq_data[i]

        # distance with other windows:
        for j in range(i + 1, total_windows):
            second_window = freq_data[j]
            # calculate distance
            dist_data[i][j] = calc_dist(current_window, second_window)
    
    # create dataframe and return dataframe
    col_names = ['d_w_' + str(i + 1) for i in range(total_windows)]
    df = pd.DataFrame(dict(zip(col_names, dist_data)))

    return df


def calc():
    
    save_path = 'data/w_51/'
    data_root_path = save_path
    dataset = 'BalSel'
    # dataset = 'AI'


    csv_files_path = os.path.join(data_root_path, 'freq', dataset)
    
    csv_files = os.listdir(csv_files_path)

    freq_dist_save_folder = os.path.join(str(save_path), 'freq_dist', str(dataset))

    files_read = 0

    for file in csv_files:
        # check if we want to read this one or not
        file_path = os.path.join(csv_files_path, file)
        
        if check_read(file_path):
            # start time 
            
            logging.info('Reading {}'.format(file_path))
            data = pd.read_csv(file_path)

            start_time = time.time()
            dist_data = calc_dist_each_col(data)
            logging.info('calc_dist_each_col time taken: {}'.format((time.time() - start_time)))

            
            start_time = time.time()
            dist_data_2 = calc_dist_no_duplicate_col(data)
            logging.info('calc_dist_no_duplicate_col time taken: {}'.format((time.time() - start_time)))


            # save dist data
            file_name = os.path.basename(file_path)
            # dist_data_file_name = 'dist_' + file_name
            # dist_data_file_loc = os.path.join(freq_dist_save_folder, dist_data_file_name)

            dist_data_file_name = 'w51_dist_' + file_name
            dist_data_file_loc = os.path.join('./debug', dist_data_file_name)

            # dist_data.to_csv(dist_data_file_loc, index=False)

            logging.info('Completed Reading {}. Saved dist info {}'.format(file_path, dist_data_file_loc))
            

            # only read few files
            files_read += 1
            if files_read == 2:
                break


def calc_dist_from_freq(
    data_root_path,
    datasets,
    save_path = '.'
):
    """
    """

    for dataset in datasets:
        logging.info('Reading freq data files in dataset {}'.format(dataset))
        # start time 
        start_time = time.time()
        
        csv_files_path = os.path.join(data_root_path, 'freq', dataset)
        csv_files = os.listdir(csv_files_path)

        # file save paths
        freq_dist_save_folder = os.path.join(str(save_path), 'freq_dist', str(dataset))
        os.makedirs(freq_dist_save_folder, exist_ok = True)

        # # this should already be created in task 1
        # plot_save_folder = os.path.join(str(save_path), 'plots', str(dataset))
        # os.makedirs(plot_save_folder, exist_ok = True)

        logging.info('Generated csv files will be saved in: {}'.format(freq_dist_save_folder))
        # logging.info('Generated plots will be saved in : {}'.format(plot_save_folder))


        for file in csv_files:
            # check if we want to read this one or not
            file_path = os.path.join(csv_files_path, file)
            if check_read(file_path):
                logging.info('Reading {}'.format(file_path))

                data = pd.read_csv(file_path)

                dist_data = calc_dist_each_col(data)

                # save dist data
                file_name = os.path.basename(file_path)
                dist_data_file_name = 'dist_' + file_name
                dist_data_file_loc = os.path.join(freq_dist_save_folder, dist_data_file_name)

                dist_data.to_csv(dist_data_file_loc, index=False)

                logging.info('Completed Reading {}. Saved dist info {}'.format(file_path, dist_data_file_loc))
                
        logging.info(
            'Reading freq files data in dataset {} took {} seconds'.format(
                dataset,
                (time.time() - start_time)
            )
        )


if __name__ == '__main__':

    # read config
    args = argument_parser()
    app_config = load_yaml_file(args.config)
    
    # create logs
    create_logfile(app_config['output_save_path'])


    logging.info('Starting application: dist from freq')
    logging.info('Reading config parameters from file {}'.format(args.config))
    logging.info(
        'Application configuration parameters: {}'.format(
            app_config
        )
    )

    calc_dist_from_freq(
        app_config['output_save_path'],
        app_config['datasets'],
        app_config['output_save_path']
    )
    
    logging.info('Application Ended Successfully')

