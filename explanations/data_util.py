import random
import numpy as np
import pandas as pd
from pandas.api import types


def get_diff_features(p, q):
    diff_features = []
    for feature in p.index:
        if p[feature] != q[feature]:
            diff_features.append(feature)
    return diff_features


def get_diff_indexes(p, q):
    diff_idx = []
    for idx, feature in enumerate(p.index):
        if q[feature] != p[feature]:
            diff_idx.append(idx)
    return diff_idx


def get_features_types(data):
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    elif isinstance(data, pd.Series):
        columns = data.index
    else:
        columns = []

    features_types = {}
    for col in columns:
        if types.is_bool_dtype(data[col]):
            features_types[col] = 'bool'
        elif types.is_categorical_dtype(data[col]):
            features_types[col] = 'categorical'
        elif types.is_integer_dtype(data[col]):
            features_types[col] = 'numeric_int'
        else:
            features_types[col] = 'numeric_float'
            
    return features_types


def sample_between(p, q, data, num_samples: int = 500):
    samples = pd.DataFrame([p] * num_samples)
    diff_features = get_diff_features(p, q)
    features_types = get_features_types(data)

    for col in diff_features:
        p_value, q_value = p[col], q[col]

        if features_types[col] == 'bool':
            samples[col] = random.choices([p_value, q_value], k=num_samples)
        elif features_types[col] == 'categorical':
            samples[col] = random.choices([p_value, q_value], k=num_samples)
        elif features_types[col] == 'numeric_int':
            low, high = sorted([p_value, q_value])
            samples[col] = np.random.randint(low, high + 1, size=num_samples)
        else:  # col type is numeric_float
            low, high = sorted([p_value, q_value])
            samples[col] = np.random.uniform(low, high, size=num_samples)

    return samples


def rate_to_html(phones_to_rate):
    # add a rating option to every cellphone
    phones_to_rate.reset_index(drop=True, inplace=True)
    for idx, _ in phones_to_rate.iterrows():
        phones_to_rate.loc[idx, 'rating'] = f'<input type="number" id="phone_{idx}" name="phone_{idx}" min="1" max="10" /> '

    # convert phone data to html code
    css_classes = 'w3-table-all w3-centered w3-hoverable w3-striped'
    phones_to_rate_html = phones_to_rate.to_html(
        classes=css_classes, 
        index=False,
        escape=False
    )
    
    return phones_to_rate_html
    
    
def phone_to_html(phone):
    css_classes = 'w3-table-all w3-centered w3-hoverable w3-striped'
    phone_to_html = phone.to_frame().T.to_html(
        classes=css_classes,
        index=False
    )
    
    return phone_to_html