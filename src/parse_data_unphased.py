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
import traceback

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



def parse_file(
    file_full_path,
    parse_only_metadata = False,
):
    """
    parse data from file
    """
    # flag_metadata = False
    flag_metadata = True
    flag_data = False

    metadata = {
        'filename': file_full_path,
        'lines': [],
        'segsites': 0
    }
    header_positions = []

    data_np = None

    with open(file_full_path) as opened_file:
        # logging.info(opened_file.readline())
        # logging.info(opened_file.readlines()[:15])
        file_lines = opened_file.readlines()
        for line in file_lines:
            if flag_data:
                # data
                data_person = list(line[:-1])
                if (data_person != []):
                    data_person_np = np.array(data_person, dtype=int)
                    # list_info.append(info_np)
                    if data_np is None:
                        data_np = data_person_np
                    else:
                        data_np = np.vstack((data_np, data_person_np))

            # for tennessen data ./ is the start of metadata.
            # for CEU data // is the start of metadata.
            # elif ('//' in line):
            elif ('//' in line or './' in line):
                # start of the file or still inside metadata
                flag_metadata = True
                metadata['lines'].append(line)
            
            elif('segsites:' in line):
                # still in meta data and we have access to segsites
                # we will only update segsites key for this. 
                # it won't be found inside metadata['lines']
                
                # # last character is return character '\n'
                # so avoiding it by line[:-1]
                # logging.info(f"{line} | {line[:-1].split(' ')[-1]}")
                # metadata['segsites'] = int(line[:-1].strip().split(' ')[-1]) 
                metadata['segsites'] = int(line[:-1].split(' ')[-1]) 

            elif('positions:' in line):
                # we have header of the data
                header_vals = (line.splitlines()[0].split('positions: ')[1]).split(' ')

                if header_vals[-1] == '':
                    # last character of the line was a space. remove it.
                    header_vals = header_vals[:-1]
                
                header_positions = np.array(header_vals, dtype=float)

                flag_data = True
                flag_metadata = False

                if parse_only_metadata:
                    break

            elif flag_metadata:
                metadata['lines'].append(line)
            
            else:
                logging.error('Some row of data without parsing boundaries set by us')

    # return parsed data
    
    # only metadata
    if parse_only_metadata:
        return metadata, header_positions, None
    
    # all info returned
    # create pandas dataframe
    if data_np is not None:
        index_values = [i for i in range(data_np.shape[0])]
        data = pd.DataFrame(
            data = data_np,  
            index = index_values,  
            columns = header_positions
        )
        
        return metadata, header_positions, data
    else:
        logging.error('error with data parsing')
        return -1



def find_center_index(header_positions, center_value = 0.5):
    """
    """
    mid_index = np.argmin(abs(header_positions - center_value))
    return mid_index


def find_l(
        header_positions,
        window_size,
        stride
        ):
    """
    figure out the max L for positions from header_positions and window_size with stride.
    """

    mid_index = find_center_index(header_positions)
    lowest_side_dist = min(mid_index, (header_positions.shape[0] - 1 - mid_index))


    return lowest_side_dist


def unique_header(df):
    """
    give each column an unique name from 0 to total column length 
    """

    total_cols = df.shape[1]
    n_cols = [str(i) for i in range(total_cols)]
    data = df.set_axis(n_cols, axis=1)
    return data, n_cols


def data_split(data, start_col, end_col):
    """
    """

    # https://stackoverflow.com/a/44736467
    split_data = data.loc[:, start_col:end_col]
    return split_data


def u_haplos_dist(u_haplos_str, max_u_haplos):
    """task3

    ref: 
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.hamming.html
    https://stackoverflow.com/questions/54172831/hamming-distance-between-two-strings-in-python
    """
    if max_u_haplos is None:
        u_haplos_distance = np.zeros(shape=(len(u_haplos_str),))
        for i in range(len(u_haplos_str)):
            s1 = list(u_haplos_str[i])
            dist = 0
            for s2 in u_haplos_str:
                dist += distance.hamming(s1, list(s2))
            u_haplos_distance[i] = dist / len(u_haplos_str)
        
        return u_haplos_distance
    else:
        u_haplos_distance = np.ones(shape=(max_u_haplos,)) * -10

        u_haplos = len(u_haplos_str)

        for i in range(max_u_haplos):
            # u_haplos_str[i] is not present so we chose value of distance to be a big negative number
            if i >= u_haplos:
                continue
            s1 = list(u_haplos_str[i])
            
            # calculate distance with all other unique haplotypes
            dist = 0
            for s2 in u_haplos_str:
                dist += distance.hamming(s1, list(s2))
            u_haplos_distance[i] = dist / (u_haplos - 1)

        return u_haplos_distance


def distance_l1_norm(s1, s2):
    n_s1 = np.array(s1, dtype=int)
    n_s2 = np.array(s2, dtype=int)

    L = len(n_s1)

    difference = n_s1 - n_s2

    l1_norm = np.sum(np.abs(difference))

    l1_norm_normalized = l1_norm / (2 * L)

    return l1_norm_normalized


def u_haplos_dist_b(u_haplos_str, max_u_haplos):
    
    u_haplos_distance = np.ones(shape=(max_u_haplos,)) * -10

    u_haplos = len(u_haplos_str)

    for i in range(max_u_haplos):
        # u_haplos_str[i + 1] is not present so we chose value of distance to be a big negative number
        if i + 1 >= u_haplos:
            break
        s1 = list(u_haplos_str[i])
        
        # calculate distance with next unique haplotype only
        s2 = list(u_haplos_str[i + 1])

        # hamming to l1 norm for unphased data
        # u_haplos_distance[i] = distance.hamming(s1, s2)
        u_haplos_distance[i] = distance_l1_norm(s1, s2)

    return u_haplos_distance


def frequency_calc(data, window_size, stride, app_config):
    """
    calculate frequency of the data with windows of window_size and stride
    """
    df_freq = []
    df_haplos_dist = []
    df_haplos_b_dist = []

    max_u_haplos = int(app_config['max_u_haplos'])

    data_cols = np.array(data.columns)
    d_index_len = data_cols.shape[0]

    # initial window start, end indexes 
    start_index = 0
    end_index = window_size + start_index - 1

    max_len = -1
    while end_index < d_index_len:
        # keep making windows and calculate frequency

        # make window data
        start_col = data_cols[start_index]
        end_col = data_cols[end_index]
        window_data = data_split(data, start_col, end_col)

        # calculate frequency
        w_d = window_data.astype('str')
        w_d['combined'] = w_d[w_d.columns].agg(''.join, axis=1)
        # freq_dist = list(w_d.value_counts('combined'))
        # freq_dist_np = np.array(w_d.value_counts('combined'))

        
        # ## task one and task three from value counts
        unique_values = w_d.value_counts('combined')
        
        # task 1
        freq_dist_np = np.array(unique_values.values)
        
        # task 3
        u_haplos_str = list(unique_values.index)
        u_haplos_distance = u_haplos_dist(u_haplos_str, max_u_haplos)
        df_haplos_dist.append(u_haplos_distance)
        # if u_haplos_distance.shape[0] > max_len:
        #     max_len = u_haplos_distance.shape[0]

        # task 3b
        u_haplos_distance_b = u_haplos_dist_b(u_haplos_str, max_u_haplos)
        df_haplos_b_dist.append(u_haplos_distance_b)

        
        # 1, 2, 3 all filled up to the original data frame size.
        # start from 1 instead of 0
        _temp_freq = [[i + 1] * val for i, val in enumerate(freq_dist_np)]
        flat_list = [
            x
            for xs in _temp_freq
            for x in xs
        ]
        df_freq.append(np.array(flat_list, dtype=int))

        # increase start_index and end_index properly
        start_index += stride
        end_index += stride
    

    # create df with 1, 2, 3 all filled up to the original data frame size.
    # create dataframe
    col_names = ['w_' + str(i + 1) for i in range(len(df_freq))]
    df = pd.DataFrame(dict(zip(col_names, df_freq)))


    # # unique haplotype distance dataframe
    # the len will not be same in two different windows....
    # we have fixed number of top haplotypes (here it is max_u_haplos) calculated because of this..
    col_names = ['w_' + str(i + 1) for i in range(len(df_haplos_dist))]
    df_haplos_distance = pd.DataFrame(dict(zip(col_names, df_haplos_dist)))

    # task 3b
    col_names = ['w_' + str(i + 1) for i in range(len(df_haplos_b_dist))]
    df_haplos_distance_b = pd.DataFrame(dict(zip(col_names, df_haplos_b_dist)))

    # return df_freq
    return df, df_haplos_distance, df_haplos_distance_b


def merge_haplos_distance(haplos_dist_l, haplos_dist_r):
    """ merge two dataframe on index
    merge is done using outer join, so all data will be available from 
    both dataframes. null values are replaced by 0.
    ref: 
    https://stackoverflow.com/questions/53645882/pandas-merging-101
    https://pandas.pydata.org/docs/reference/api/pandas.merge.html#pandas.merge
    """
    l_shape = haplos_dist_l.shape
    r_shape = haplos_dist_r.shape

    haplos_dist_l['index_key'] = [i for i in range(l_shape[0])]
    haplos_dist_r['index_key'] = [i for i in range(r_shape[0])]


    merge_df = haplos_dist_l.merge(haplos_dist_r, on='index_key', how='outer')

    # replace the null values with zero
    merge_df = merge_df.fillna(0)

    # remove the index_key column
    merge_df = merge_df.drop(['index_key'], axis=1)

    # change the column names
    col_names = ['w_' + str(i + 1) for i in range(merge_df.shape[1])]
    merge_df.columns = col_names


    return merge_df


# ---------------------------------------- xxxx ----------------------------------------
def find_lowest_segsites(
    data_root_path, 
    datasets, 
    window_size, 
    stride
):
    lowest_segsites_from_center = {
        'value': 100000,
        'filename': ''
    }

    for dataset in datasets:
        logging.info('Reading metadata for all files in dataset {}'.format(dataset))
        # start time 
        start_time = time.time()
        # list all the files in one dataset folder
        dataset_path = os.path.join(data_root_path, dataset)
        dataset_files = os.listdir(dataset_path)

        # check all files are .ms or not
        non_ms_files = [f for f in dataset_files if '.ms' not in f]
        assert len(non_ms_files) == 0, logging.error('Non .ms files are in {}'.format(dataset_path))

        # report found files description
        logging.info('Total files found in {} = {}'.format(dataset_path, len(dataset_files)))


        # start parcing each files in dataset to figure out min segsites
        
        # find the lowest segsites and L for all datasets
        for file in dataset_files:
            file_full_path = os.path.join(dataset_path, file)
            # file_full_path = os.path.join(dataset_path, dataset_files[0])
            
            
            metadata, header_positions, _ = parse_file(file_full_path, parse_only_metadata=True)

            _lowest_side_len = find_l(header_positions, window_size=window_size, stride=stride)

            if _lowest_side_len < lowest_segsites_from_center['value']:
                lowest_segsites_from_center['value'] = _lowest_side_len
                lowest_segsites_from_center['filename'] = metadata['filename']
        
        logging.info(
            'Reading all files metadata in dataset {} took {} seconds'.format(
                dataset,
                (time.time() - start_time)
            )
        )
    
    return lowest_segsites_from_center


def find_lowest_segsites_4_all_files(
    data_root_path, 
    datasets, 
    window_size, 
    stride
):
    lowest_segsites_from_center = {
        'value': 100000,
        'filename': ''
    }

    df_segsts = pd.DataFrame(columns = ['dataset', 'filename', 'segsite'])

    for dataset in datasets:
        logging.info('Reading metadata for all files in dataset {}'.format(dataset))
        # start time 
        start_time = time.time()
        # list all the files in one dataset folder
        dataset_path = os.path.join(data_root_path, dataset)
        dataset_files = os.listdir(dataset_path)

        # check all files are .ms or not
        non_ms_files = [f for f in dataset_files if '.ms' not in f]
        assert len(non_ms_files) == 0, logging.error('Non .ms files are in {}'.format(dataset_path))

        # report found files description
        logging.info('Total files found in {} = {}'.format(dataset_path, len(dataset_files)))


        # start parcing each files in dataset to figure out min segsites
        
        # find the lowest segsites and L for all datasets
        for file in dataset_files:
            file_full_path = os.path.join(dataset_path, file)
            # file_full_path = os.path.join(dataset_path, dataset_files[0])
            
            
            metadata, header_positions, _ = parse_file(file_full_path, parse_only_metadata=True)

            
            _lowest_side_len = find_l(header_positions, window_size=window_size, stride=stride)

            temp_data = {
                'dataset': dataset,
                'filename': file,
                'segsite': _lowest_side_len
            }

            df_segsts = pd.concat([pd.DataFrame([temp_data]), df_segsts], ignore_index=True)

            if _lowest_side_len < lowest_segsites_from_center['value']:
                lowest_segsites_from_center['value'] = _lowest_side_len
                lowest_segsites_from_center['filename'] = metadata['filename']
        
        logging.info(
            'Reading all files metadata in dataset {} took {} seconds'.format(
                dataset,
                (time.time() - start_time)
            )
        )
    
    return lowest_segsites_from_center, df_segsts


def check_lowest_segsites_cover_bound(
        data_root_path, 
        datasets, 
        window_size, 
        stride, 
        mb_threshold = 0.25
    ):


    lowest_segsites_from_center = {
        'value': 1557,
        'filename': ''
    }

    df_segsts = pd.DataFrame(columns = ['dataset', 'filename', 'l_start_col'])
    missed_threshold_files_count = 0

    for dataset in datasets:
        logging.info('Reading metadata for all files in dataset {}'.format(dataset))
        # start time 
        start_time = time.time()
        # list all the files in one dataset folder
        dataset_path = os.path.join(data_root_path, dataset)
        dataset_files = os.listdir(dataset_path)

        # check all files are .ms or not
        non_ms_files = [f for f in dataset_files if '.ms' not in f]
        assert len(non_ms_files) == 0, logging.error('Non .ms files are in {}'.format(dataset_path))

        # report found files description
        logging.info('Total files found in {} = {}'.format(dataset_path, len(dataset_files)))


        for file in dataset_files:
            file_full_path = os.path.join(dataset_path, file)
            # file_full_path = os.path.join(dataset_path, dataset_files[0])
            
            
            metadata, header_positions, _ = parse_file(file_full_path, parse_only_metadata=True)

            index_center = find_center_index(header_positions)
            index_start = index_center - lowest_segsites_from_center['value']

            # assert index_start >= 0, logging.error('index start is less than 0 for file: {}'.format(file_full_path))
            if index_start >= 0:
                start_col = header_positions[index_start]

                temp_data = {
                    'dataset': dataset,
                    'filename': file,
                    'l_start_col': start_col
                }

                df_segsts = pd.concat([pd.DataFrame([temp_data]), df_segsts], ignore_index=True)

                if start_col > mb_threshold:
                    # it does not cover the mb threshold in SNPs
                    logging.error(f'{file} does not cover {mb_threshold} towards the left side. covered position: {start_col}')
                    missed_threshold_files_count += 1
            else:
                logging.error('index start is less than 0 for file: {}'.format(file_full_path))
                temp_data = {
                    'dataset': dataset,
                    'filename': file,
                    'l_start_col': -1
                }

                df_segsts = pd.concat([pd.DataFrame([temp_data]), df_segsts], ignore_index=True)
                missed_threshold_files_count += 1


        
        logging.info(
            'Reading all files metadata in dataset {} took {} seconds'.format(
                dataset,
                (time.time() - start_time)
            )
        )
    
    logging.info('Missed threshold files count: {}'.format(missed_threshold_files_count))

    return df_segsts, missed_threshold_files_count


# ---------------------------------------- xxxx ----------------------------------------




# ---------------------------------------- xxxx ----------------------------------------
def generate_window_frequency(
    app_config,
    data_root_path, 
    datasets, 
    window_size, 
    stride, 
    lowest_segsites_from_center,
    save_path = '.'
):

    logging.info(
        'Calculating frequency on datasets: {} with lowest_segsites_from_center: {}'.format(
            datasets,
            lowest_segsites_from_center
        )
    )

    for dataset in datasets:
        logging.info('Reading data for all files in dataset {}'.format(dataset))
        # start time 
        start_time = time.time()
        # list all the files in one dataset folder
        dataset_path = os.path.join(data_root_path, dataset)
        dataset_files = os.listdir(dataset_path)

        # for each dataset, we will merge all the frequnecy dataframes in a dataframe as avg. freq.
        file_count = 0
        all_data = None

        # task1 file save paths
        freq_save_folder = os.path.join(str(save_path), 'freq', str(dataset))
        os.makedirs(freq_save_folder, exist_ok = True)

        # task3 file save paths
        h_dist_save_folder = os.path.join(str(save_path), 'w_haplo_d', str(dataset))
        os.makedirs(h_dist_save_folder, exist_ok = True)

        # task3b file save paths
        h_dist_b_save_folder = os.path.join(str(save_path), 'w_haplo_d_b', str(dataset))
        os.makedirs(h_dist_b_save_folder, exist_ok = True)

        plot_save_folder = os.path.join(str(save_path), 'plots', str(dataset))
        os.makedirs(plot_save_folder, exist_ok = True)

        logging.info('Generated csv files will be saved in: {}'.format(freq_save_folder))
        logging.info('Generated haplotype distances will be saved in: {}'.format(h_dist_save_folder))
        logging.info('Generated plots will be saved in : {}'.format(plot_save_folder))
        

        # start parcing each files in dataset
        
        # find the frequency for each file
        for file in dataset_files:
            file_full_path = os.path.join(dataset_path, file)
            # file_full_path = os.path.join(dataset_path, dataset_files[0])
            
            # get the data for the file
            metadata, header_positions, data = parse_file(file_full_path, parse_only_metadata=False)
            # ds.append(data)

            # calculate index_start, index_center, index_end
            index_center = find_center_index(header_positions)
            index_start = index_center - lowest_segsites_from_center['value']
            index_end = index_center + lowest_segsites_from_center['value']

            assert index_start >= 0, logging.error('index start is less than 0 for file: {}'.format(file_full_path))
            assert index_end <= header_positions.shape[0] - 1, logging.error('index end is larger than all columns in file {}'.format(file_full_path))


            ## bug fix for same header val for different positions.
            data, header_positions = unique_header(data)

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


            # save heatmap -> instead of saving for each file, save for each database
            ## --------- save dataframe
            # create directory
            # freq_save_folder = os.path.join(str(data_root_path), 'freq', str(dataset))
            # os.makedirs(freq_save_folder, exist_ok = True)

            # task 1 dataframe
            # create filepath
            file_name = 'freq_' + file.replace('.ms', '.csv')
            freq_file_path = os.path.join(freq_save_folder, file_name)
            # dump csv
            m_freq.to_csv(freq_file_path, index=False)

            # task 3 dataframe
            file_name = 'haplo_d_' + file.replace('.ms', '.csv')
            haplo_dist_file_path = os.path.join(h_dist_save_folder, file_name)
            # dump csv
            m_haplos_dist.to_csv(haplo_dist_file_path, index=False)
            ## --------- save dataframe

            # task 3b dataframe
            file_name = 'haplo_d_b_' + file.replace('.ms', '.csv')
            haplo_dist_file_path = os.path.join(h_dist_b_save_folder, file_name)
            # dump csv
            m_haplos_dist_b.to_csv(haplo_dist_file_path, index=False)
            ## --------- save dataframe

            # add data
            if all_data is None:
                all_data = m_freq
            else:
                all_data += m_freq

            file_count += 1
            logging.info(
                '{} complete. Total completed in dataset: {}'.format(
                    file,
                    file_count
                )
            )
            
        
        # make the avg data
        if file_count:
            all_data /= file_count
            ## ----- save avg data
            # create directory
            # freq_save_folder = os.path.join(str(data_root_path), 'freq', str(dataset))
            # os.makedirs(freq_save_folder, exist_ok = True)
            # create filepath
            file_name = 'all_freq_avg_' + str(dataset) + '.csv'
            freq_file_path = os.path.join(freq_save_folder, file_name)
            # dump csv
            all_data.to_csv(freq_file_path, index=False)
            ## ----- save avg data

            ## ----- save heatmap
            v_max_df = max(all_data.iloc[-1,:])
            cp = sns.color_palette("tab10", n_colors=int(v_max_df))
            title = 'all_freq_avg_{}'.format(dataset)
            # ax = sns.heatmap(all_data)
            ax = sns.heatmap(
                data=all_data, 
                cmap=cp, 
                vmin=0, 
                vmax=v_max_df, 
                yticklabels=False, 
                xticklabels=False)
            ax.set_title(title)
            
            file_name = str(title) + '.png'
            plot_full_path = os.path.join(plot_save_folder, file_name)
            # will save in this directory
            plt.savefig(plot_full_path)
            plt.close()
            ## ----- save heatmap

        else:
            logging.error('File count is 0. No file could be read!!')
        
        logging.info(
            'Reading all files data in dataset {} took {} seconds'.format(
                dataset,
                (time.time() - start_time)
            )
        )

# ---------------------------------------- xxxx ----------------------------------------


def main(app_config):
    try:
        lowest_segsites_from_center = find_lowest_segsites(
            app_config['data_root_path'], 
            app_config['datasets'], 
            app_config['window_size'], 
            app_config['stride']
        )
        logging.info('Lowest segsites: {}'.format(lowest_segsites_from_center))
        
        # max segsites we will use
        logging.info('Using Lowest segsites: {}'.format(lowest_segsites_from_center))

        generate_window_frequency(
            app_config, 
            app_config['data_root_path'], 
            app_config['datasets'], 
            app_config['window_size'], 
            app_config['stride'],
            lowest_segsites_from_center,
            save_path=app_config['output_save_path']
        )

    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)
        logging.error(repr(traceback.extract_tb(e.__traceback__)))


def main_debug(app_config):
    try:

        if True:
            lowest_segsites_from_center, df = find_lowest_segsites_4_all_files(
                app_config['data_root_path'], 
                app_config['datasets'], 
                app_config['window_size'], 
                app_config['stride']
            )
            file_location = app_config['output_save_path'] + 'segsites.csv'
            df.to_csv(file_location, index=False)
            # max segsites we will use
            logging.info('Lowest segsites: {}'.format(lowest_segsites_from_center))


        if True:
            df, missed_count = check_lowest_segsites_cover_bound(
                app_config['data_root_path'], 
                app_config['datasets'], 
                app_config['window_size'], 
                app_config['stride'], 
                # 0.25 # 0.5mb,
                0.5 # 0.8mb
            )
            file_location = app_config['output_save_path'] + 'lowest_segsites_cover_bound.csv'
            df.to_csv(file_location, index=False)
            # max segsites we will use
            logging.info('Missed count: {}'.format(missed_count))

        
    except Exception as e:
        logging.error("An exception occurred")
        logging.error(e)
        logging.error(repr(traceback.extract_tb(e.__traceback__)))



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

    main(app_config=app_config)
    

    logging.info('Application Ended Successfully')

