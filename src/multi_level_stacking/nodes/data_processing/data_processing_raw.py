import pandas as pd
import yaml
from sklearn.preprocessing import binarize
import numpy as np


def preprocess_raw_data(raw_data, conf):
    steps_list = conf['steps']
    if 'remove_columns' in steps_list:
        raw_data = _remove_columns(raw_data, conf['columns_to_remove'])
    if 'rename_columns' in steps_list:
        raw_data = _define_col_names(raw_data, conf['col_names_list'])
    if 'target_replace' in steps_list:
        raw_data = _target_replace(raw_data, conf['target']['col'], conf['target']['map'])
    if 'replace_missing_value' in steps_list:
        raw_data = _replace_missing_value(raw_data, conf['missing_value'])
        raw_data.dropna(inplace=True)
    if 'replace_string_columns' in steps_list:
        raw_data = _replace_string_columns(raw_data)

    return raw_data

def _define_col_names(data, col_names_list):
    data = data.set_axis(col_names_list, axis=1)

    return data

def _target_replace(data,
                    target_col_name,
                    map):
    
    data[target_col_name] = data[target_col_name].replace(map)

    return data

def _replace_missing_value(data, missing_value):
    data = data.replace(missing_value, np.nan)
    return data

def _remove_columns(data, columns_to_remove):
    data = data.drop(columns=columns_to_remove)
    return data

def _replace_string_columns(data):
    string_columns = data.select_dtypes(include=['object']).columns
    for col in string_columns:
        data[col] = data[col].astype('category').cat.codes
    return data