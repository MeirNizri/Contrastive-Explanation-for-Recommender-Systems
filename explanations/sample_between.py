import random
import numpy as np
import pandas as pd
from pandas.api import types


def get_diff_features(p, q):
    unexplainable_features = ['time from release', 'model']
    diff_features, diff_idx = ([], [])

    for idx, feature in enumerate(p.index):
        if q[feature] != p[feature] and feature not in unexplainable_features:
            diff_features.append(feature)
            diff_idx.append(idx)
    return diff_features, diff_idx


def get_features_types(data) -> list:
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    elif isinstance(data, pd.Series):
        columns = data.index

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


def sample_between(p, q, num_samples: int = 1000):
    samples = pd.DataFrame([p] * num_samples)
    diff_features, _ = get_diff_features(p, q)
    features_types = get_features_types(p)

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
