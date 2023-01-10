import pandas as pd
import numpy as np

from Raw_Data.raw_data_consts import TIMESTAMP, RIGHT_PUPIL, LEFT_PUPIL
from Raw_Data.tracker_data.read_trackers import read_tracker
from Raw_Data.tracker_data.filter_trials import filter_data, filter_short_trials
from Raw_Data.tracker_data.intepolate import data_interpolation
from Raw_Data.utils.timestamp_correction import timestamp_correction
from Raw_Data.utils.indices_of_interest import idx_of_return, idx_of_start, idx_of_back
import Raw_Data.configurations as cfg


def reset_index(data):
    for (_, df) in data:
        df.reset_index(inplace=True, drop=True)

    return data


def choose_relevent_parts(data, mode="all", ts_name='Hand_loc_Y'):
    # define filter function based on the chosen mode
    if mode == "all":
        fun = lambda x: x
    elif mode == "before":
        fun = lambda x: x.iloc[:idx_of_start(x[ts_name])]
    elif mode == "movement":
        fun = lambda x: x.iloc[idx_of_start(x[ts_name]):idx_of_back(x[ts_name])]
    elif mode == "reach":
        fun = lambda x: x.iloc[idx_of_start(x[ts_name]):idx_of_return(x[ts_name])]
    elif mode == "return":
        fun = lambda x: x.iloc[idx_of_return(x[ts_name]):idx_of_back(x[ts_name])]

    for i, (_, df) in enumerate(data):
        data[i] = (data[i][0], fun(df))

    return data


def no_pupil_frame_filter(df):
    output_cols = df.columns

    df['is_blink'] = (df[RIGHT_PUPIL] == -1) | (df[LEFT_PUPIL] == -1)
    df['blink_diff'] = df['is_blink'] - df['is_blink'].shift(1, fill_value=0)
    df['blink_diff_shift'] = df['blink_diff'].shift(-1, fill_value=0)
    df['blink_diff'] = np.where((df['blink_diff'] == -1) | (df['blink_diff_shift'] == -1),
                                df['blink_diff_shift'], df['blink_diff'])

    blink_df = df[[TIMESTAMP, 'blink_diff']]
    blink_df_start = blink_df[blink_df['blink_diff'] == 1]
    blink_df_end = blink_df[blink_df['blink_diff'] == -1]
    blink_df['blink_diff'] = 0

    blink_df_start[TIMESTAMP] -= cfg.blink_window
    blink_df_end[TIMESTAMP] = cfg.blink_window

    blink_df = pd.concat([blink_df_start, blink_df_end, blink_df])
    blink_df = blink_df.sort_values(TIMESTAMP)
    blink_df['blink_cum_sum'] = blink_df['blink_diff']
    blink_df['is_blink'] = (blink_df['blink_cum_sum'] > 0).astype(int)

    blink_df = blink_df[[TIMESTAMP, 'is_blink']]

    df = df.merge(blink_df, on='timestamp', how='inner')

    df = df[df['is_blink'] == 0]

    df = df[output_cols]
    return df


def no_pupil_frame_filter_all(data):
    for i in range(len(data)):
        df = data[i][1]
        data[i] = (data[i][0], no_pupil_frame_filter(df))

    return data


def baseline_normalization(df):
    start_time = df.loc[TIMESTAMP, 0]
    trial_beginning = df[df[TIMESTAMP] < start_time + cfg.normalization_window]

    right_pupil_median = trial_beginning[RIGHT_PUPIL].median()
    left_pupil_median = trial_beginning[LEFT_PUPIL].median()

    df[RIGHT_PUPIL] -= right_pupil_median
    df[LEFT_PUPIL] -= left_pupil_median

    return df


def baseline_normalization_all(data):
    for i in range(len(data)):
        df = data[i][1]
        data[i] = (data[i][0], baseline_normalization(df))

    return data


def timestamp_to_ms(data):
    for (_, df) in data:
        df['timestamp'] = timestamp_correction(df['timestamp'])

    return data


def timestamp_from_zero(data):
    for (_, df) in data:
        zero = df['timestamp'].iat[0]
        df['timestamp'] = df['timestamp'] - zero

    return data


def drop_extra(data):
    for (_, df) in data:
        df.drop(labels=cfg.to_drop, axis=1, inplace=True)

    return data


def cut_signal(data):
    for i in range(len(data)):
        data[i] = (data[i][0], data[i][1][data[i][1]['timestamp'] > cfg.start_signal])

    return data


def tracker_preprocessing(subject_num):
    # read subject data
    data = read_tracker(subject_num)

    # reset dataframe index
    data = reset_index(data)

    # format timestamp in miliseconds
    data = timestamp_to_ms(data)

    # if the signal is pupil dimeteter, throw rows with -1 values
    if cfg.pathes.trial_mode.startswith('pupil'):
        data = baseline_normalization_all(data)
        data = no_pupil_frame_filter_all(data)

    # filter trials
    data = filter_data(data)

    # choose relevant part of the trial
    data = choose_relevent_parts(data, mode='all')

    # filter trials with to little amount of data left
    data = filter_short_trials(data)

    # drop extra columns
    if cfg.to_drop:
        data = drop_extra(data)

    # reset dataframe index
    data = reset_index(data)

    # start any trial's timestamp from zero
    data = timestamp_from_zero(data)

    # interpolate the data in order to hve even space between frames
    data = data_interpolation(data)

    # cut the begining of the signal
    if cfg.start_signal > 0:
        data = cut_signal(data)
        data = reset_index(data)
        data = timestamp_to_ms(data)

    return data
