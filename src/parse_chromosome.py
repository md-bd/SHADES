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

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial import distance


sys.path.append('.')
from src.parse_data import data_split, frequency_calc, merge_haplos_distance
from src.dist_from_freq import calc_dist_each_col

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


def check_metadata(line):
    if line.startswith("##"):
        return True
    else:
        return False


def read_data(line):

    try:
        all_data = line.split()
        position = all_data[1]
        data = []
        meta_data = []
        for d in all_data:
            if '|' in d:
                # temp = d.split("|")
                temp = [int(x) for x in d.split("|")]
                data.extend(temp)
            else:
                meta_data.append(d)
        
        return data, meta_data
    
    except Exception as e:
        logging.error(e)
        sys.exit(-1)
    

def find_center_position(data):
    col = data.columns

    if len(col) % 2 == 0:
        # even
        logging.error('Center position is found to be even! It is not suppposed to happen')
        center = len(col) // 2
        pos = (int(col[center]) + int(col[center - 1])) / 2
    else:
        # odd
        center = len(col) // 2
        pos = int(col[center])
    
    return pos


def find_center_index(header_positions):
    header_len = len(header_positions)
    if header_len % 2 == 1:
        # odd
        center_pos = header_len // 2
    else:
        # even
        logging.error('header position is even! This should not happen')
        center_pos = header_len // 2
    return center_pos


def task_1_3(
    data,
    header_positions,
    window_size,
    stride,
    lowest_segsites_from_center
):
    # calculate index_start, index_center, index_end
    index_center = find_center_index(header_positions) # write this function differently for chromosome
    index_start = index_center - lowest_segsites_from_center['value']
    index_end = index_center + lowest_segsites_from_center['value']

    assert index_start >= 0, logging.error('index start is less than 0 for data')
    assert index_end <= header_positions.shape[0] - 1, logging.error('index end is larger than all columns in data')

    # split the data into two sides
    # left side
    start_col = header_positions[index_start]
    end_col = header_positions[int(index_center + (window_size - 1)/2 - stride)]
    data_l = data_split(data, start_col, end_col)

    # right side
    start_col = header_positions[int(index_center - (window_size - 1)/2)]
    end_col = header_positions[index_end]
    data_r = data_split(data, start_col, end_col)


    # frequency from window size and stride
    freq_l, haplos_dist_l, haplos_dist_b_l = frequency_calc(data_l, window_size, stride, app_config)
    freq_r, haplos_dist_r, haplos_dist_b_r = frequency_calc(data_r, window_size, stride, app_config)

    # join both frequency values and reassign column names
    m_freq = freq_l.join(freq_r, lsuffix='_l', rsuffix='_r')
    col_names = ['w_' + str(i + 1) for i in range(m_freq.shape[1])]
    m_freq.columns = col_names

    # join both haplos_dist 
    # l and r will not have same amount of rows...
    m_haplos_dist = merge_haplos_distance(haplos_dist_l, haplos_dist_r)

    # task 3b
    m_haplos_dist_b = merge_haplos_distance(haplos_dist_b_l, haplos_dist_b_r)


    return m_freq, m_haplos_dist, m_haplos_dist_b


# ---------------------------------------- xxxx ----------------------------------------
def generate_window_frequency(
    app_config,
    data_root_path, 
    chromosomes, 
    window_size, 
    stride, 
    lowest_segsites_from_center,
    save_path = '.'
):

    logging.info(
        'Calculating frequency on chromosomes: {} with lowest_segsites_from_center: {}'.format(
            chromosomes,
            lowest_segsites_from_center
        )
    )

    position_stride = 0
    if 'position_stride' in app_config and app_config['position_stride'] != '':
        position_stride = int(app_config['position_stride'])
    else:
        position_stride = 21
        logging.warning('using default position_stride')

    logging.info('position stride is {}'.format(position_stride))

    for chromosome in chromosomes:
        logging.info('Reading data for file in chromosome {}'.format(chromosome))
        # start time 
        start_time = time.time()
        # list all the files in one dataset folder
        dataset_path = os.path.join(data_root_path, chromosome)

        # for each dataset, we will merge all the frequnecy dataframes in a dataframe as avg. freq.
        file_count = 0
        # all_data = None

        # chromosome to .ms file shape data
        ms_like_save_folder = os.path.join(str(save_path), 'ms_data', str(chromosome))
        os.makedirs(ms_like_save_folder, exist_ok = True)

        # task1 file save paths
        freq_save_folder = os.path.join(str(save_path), 'freq', str(chromosome))
        os.makedirs(freq_save_folder, exist_ok = True)

        # task3 file save paths
        h_dist_save_folder = os.path.join(str(save_path), 'w_haplo_d', str(chromosome))
        os.makedirs(h_dist_save_folder, exist_ok = True)

        # task3b file save paths
        h_dist_b_save_folder = os.path.join(str(save_path), 'w_haplo_d_b', str(chromosome))
        os.makedirs(h_dist_b_save_folder, exist_ok = True)

        plot_save_folder = os.path.join(str(save_path), 'plots', str(chromosome))
        os.makedirs(plot_save_folder, exist_ok = True)

        # file save paths
        freq_dist_save_folder = os.path.join(str(save_path), 'freq_dist', str(chromosome))
        os.makedirs(freq_dist_save_folder, exist_ok = True)

        logging.info('Parsed data similar to each ms file will be saved in : {}'.format(ms_like_save_folder))
        logging.info('Generated frequency csv files will be saved in: {}'.format(freq_save_folder))
        logging.info('Generated csv files will be saved in: {}'.format(freq_dist_save_folder))
        logging.info('Generated haplotype distances will be saved in: {}'.format(h_dist_save_folder))
        logging.info('Generated plots will be saved in : {}'.format(plot_save_folder))
        

        # start parcing chromosome file
        # find the frequency for dataset_file
        file = chromosome + '.vcf'
        file_full_path = os.path.join(dataset_path, file)

        max_segsite = int(lowest_segsites_from_center['value'])
        # window_size = 21
        all_metadata_read = False

        max_segsites_for_each_center_position = (max_segsite * 2) + 1

        with open(file_full_path) as data_file:
            
            rows_read = 0
            data_df = pd.DataFrame()

            for i, line in enumerate(data_file):
                if all_metadata_read:
                    # read data
                    l_data, l_mdata = read_data(line)
                    position = l_mdata[1]

                    # same position multiple times occurance
                    if position in data_df.columns:
                        logging.warning(f'{position} position is previously read in {chromosome}. skipping this position')
                        logging.warning('delete this position manually')
                        data_df = data_df.drop(columns=[position])
                        rows_read -= 1

                        continue

                    data_df[position] = l_data
                    rows_read += 1

                    if rows_read == max_segsites_for_each_center_position:

                        logging.info(f'one segsite read: {i} | last position read: {position}')
                        # rows_read = 0
                        # rows_read = rows_read - window_size
                        rows_read = rows_read - position_stride

                        # find the center position
                        center_pos = find_center_position(data_df)
                        filename = str(chromosome) + "_" + str(int(center_pos)) + ".csv"
                        
                        # save to a file
                        logging.info(f'{data_df.shape}, {filename}')
                        # logging.info(data_df.head())
                        ms_like_filepath = os.path.join(ms_like_save_folder, filename)
                        data_df.to_csv(ms_like_filepath, index=False)
                        logging.info(f'ms like file saved at: {ms_like_filepath}')

                        # do post processing
                        cols = data_df.columns
                            
                        
                        m_freq, m_haplos_dist, m_haplos_dist_b = task_1_3(
                            data_df,
                            cols,
                            window_size,
                            stride,
                            lowest_segsites_from_center
                        )

                        dist_data = calc_dist_each_col(m_freq)
                        
                        ## --------- save dataframe
                        # task 1 dataframe
                        # create filepath
                        file_name = 'freq_' + filename
                        freq_file_path = os.path.join(freq_save_folder, file_name)
                        # dump csv
                        m_freq.to_csv(freq_file_path, index=False)

                        # task 3 dataframe
                        file_name = 'haplo_d_' + filename
                        haplo_dist_file_path = os.path.join(h_dist_save_folder, file_name)
                        # dump csv
                        m_haplos_dist.to_csv(haplo_dist_file_path, index=False)
                        

                        # task 3b dataframe
                        file_name = 'haplo_d_b_' + filename
                        haplo_dist_file_path = os.path.join(h_dist_b_save_folder, file_name)
                        # dump csv
                        m_haplos_dist_b.to_csv(haplo_dist_file_path, index=False)

                        # task 2
                        dist_data_file_name = 'dist_' + filename
                        dist_data_file_loc = os.path.join(freq_dist_save_folder, dist_data_file_name)
                        # dump csv
                        dist_data.to_csv(dist_data_file_loc, index=False)
                        ## --------- save dataframe
                        
                        
                        # # remove the first window_size columns and keep the rest
                        # keep_cols = cols[window_size:]
                        # data_df = data_df[keep_cols]
                        # logging.info('after removing window_size columns from the front, dataframe shape: {}'.format(data_df.shape))

                        # remove the first position_stride columns and keep the rest
                        keep_cols = cols[position_stride:]
                        data_df = data_df[keep_cols]
                        logging.info('after removing position_stride({}) columns from the front, dataframe shape: {}'.format(position_stride, data_df.shape))

                        file_count += 1
                        logging.info(
                            'Total completed in {}: {}'.format(
                                chromosome, 
                                file_count
                            )
                        )
                        
                else:
                    # read metadata
                    metadata_flag = check_metadata(line)
                    if not metadata_flag:
                        all_metadata_read = True
                        # print(f"i: {i} line: {line}")
                        # this is the header line
                        header = line[1:].split()
                        print(header)
                        print(i)

        
        logging.info(
            'Reading file data in chromosome {} took {} seconds'.format(
                chromosome,
                (time.time() - start_time)
            )
        )

# ---------------------------------------- xxxx ----------------------------------------


if __name__ == '__main__':

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

    lowest_segsites_from_center = app_config['lowest_segsites_from_center']

    # max segsites we will use
    logging.info('Lowest segsites: {}'.format(lowest_segsites_from_center))

    generate_window_frequency(
        app_config, 
        app_config['data_root_path'], 
        app_config['chromosomes'], 
        app_config['window_size'], 
        app_config['stride'],
        lowest_segsites_from_center,
        save_path=app_config['output_save_path']
    )


    logging.info('Application Ended Successfully')

