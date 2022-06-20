import random
import numpy as np
import pandas as pd
from pandas.api import types
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .sample_between import sample_between, get_diff_features


def contrast_exp(rs_model, user_id, p, q):
    # get p and q ratings from the recommendation model, p should be better than q
    p_copy, q_copy = (p.copy(), q.copy())
    items = pd.concat([p_copy, q_copy], axis=1).T
    p_rating, q_rating = rs_model.predict(items, user_id)
    if q_rating > p_rating:
        temp = (p_copy, p_rating)
        p_copy, p_rating = (q_copy, q_rating)
        q_copy, q_rating = temp

    # sample uniformly objects from the items' data that are "between" p and q
    num_samples = 500
    x_data = sample_between(p_copy, q_copy, num_samples)
    x_data.drop_duplicates(inplace=True)
    x_data.reset_index(drop=True, inplace=True)
    y_data = pd.DataFrame(rs_model.predict(x_data, user_id), columns=['prediction'])
    # clear samples with all zeros
    null_index = y_data.loc[y_data['prediction'].isnull()].index.tolist()
    x_data = x_data.drop(null_index)
    y_data = y_data.drop(null_index)

    # fit linear regression model and get weights for every feature
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(x_data, y_data)
    weights = linear_regression_model.coef_[0]

    # get the scaled weights by w*(p-q) and save only features with a difference between p and q
    scaled_weights = np.multiply(weights, np.subtract(p_copy.to_numpy(), q_copy.to_numpy()))
    _, diff_idx = get_diff_features(p_copy, q_copy)
    scaled_weights = scaled_weights[diff_idx]
    features = np.array(p.index)[diff_idx]

    # integrate between features and weights and sort from high to low
    features_weights = dict(zip(features, scaled_weights))
    features_weights = sorted(features_weights.items(), key=lambda x: x[1], reverse=True)
    features_weights = dict(features_weights)
    try:
        del features_weights['Time from release']
    except Exception:
        pass

    # build contrastive explanation by iterating the features in order from high weight to low
    explanation = []
    for feature in features_weights.keys():
        if q_rating >= p_rating or features_weights[feature] <= 0:
            break
        explanation.append(feature)
        # Update one feature of q to be as in p and check if the rating of q is higher than p
        q_copy[feature] = p_copy[feature]
        items = pd.concat([p_copy, q_copy], axis=1).T
        _, q_rating = rs_model.predict(items, user_id)

    return explanation


def random_contrast_exp(rs_model, user_id, p, q):
    # get p and q ratings from the recommendation model, p should be better than q
    p_copy, q_copy = (p.copy(), q.copy())
    items = pd.concat([p_copy, q_copy], axis=1).T
    p_rating, q_rating = rs_model.predict(items, user_id)
    if q_rating > p_rating:
        temp = (p_copy, p_rating)
        p_copy, p_rating = (q_copy, q_rating)
        q_copy, q_rating = temp

    # randomly shuffle all the different features between p and q
    diff_features, _ = get_diff_features(p_copy, q_copy)
    try:
        diff_features.remove('Time from release')
    except Exception:
        pass
    random.shuffle(diff_features)

    # build contrastive explanation by iterating the features
    explanation = []
    for feature in diff_features:
        if q_rating >= p_rating:
            break
        explanation.append(feature)
        # Update one feature of q to be as in p and check if the rating of q is higher than p
        q_copy[feature] = p_copy[feature]
        items = pd.concat([p_copy, q_copy], axis=1).T
        _, q_rating = rs_model.predict(items, user_id)

    return explanation


def brute_contrast_exp(rs_model, user_id, p, q):
    # get p and q ratings from the recommendation model, p should be better than q
    p_copy, q_copy = (p.copy(), q.copy())
    items = pd.concat([p_copy, q_copy], axis=1).T
    p_rating, q_rating = rs_model.predict(items, user_id)
    if q_rating == p_rating:
        return []
    if q_rating > p_rating:
        temp = (p_copy, p_rating)
        p_copy, p_rating = (q_copy, q_rating)
        q_copy, q_rating = temp

    # get all the different features between p and q
    diff_features, _ = get_diff_features(p_copy, q_copy)
    try:
        diff_features.remove('Time from release')
    except Exception:
        pass

    # iterate on every combination of diff_features by length
    for i in range(1, len(diff_features)):
        for features in combinations(diff_features, i):
            features = list(features)
            # replace features of q to be as in p and check if the rating of q is higher than p
            temp_q = q_copy.copy()
            temp_q[features] = p_copy[features]
            items = pd.concat([p_copy, temp_q], axis=1).T
            _, q_rating = rs_model.predict(items, user_id)

            if q_rating >= p_rating:
                return features

    return diff_features


def lr_contrast_exp(user_data, user_ratings):
    # normalize data
    x_data = user_data.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)
    normalize_user_data = pd.DataFrame(x_scaled)

    # fit linear regression model and get weights for every feature
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(normalize_user_data, user_ratings)
    weights = linear_regression_model.coef_

    # integrate between features and weights and sort from high to low
    features = np.array(user_data.columns)
    features_weights = dict(zip(features, weights))
    features_weights = sorted(features_weights.items(), key=lambda x: x[1], reverse=True)
    sorted_features = list(map(lambda x: x[0], features_weights))
    sorted_features.remove('Time from release')

    # Return the three features with the highest weight
    return sorted_features[:3]


def test_contrast_exp(p, q, max_len=3):
    # randomly shuffle all the different features between p and q
    diff_features, _ = get_diff_features(p, q)
    try:
        diff_features.remove('Time from release')
    except Exception:
        pass
    random.shuffle(diff_features)

    diff = ""
    for feature in diff_features[:max_len]:
        if types.is_categorical_dtype(p[feature]):
            diff += f'The recommended item {feature} is {q[feature]} ' \
                    f'and the item offered by the user is {p[feature]}.<br>'
        else:
            diff += f'The recommended item has a {feature} of {q[feature]} ' \
                    f'compared to {p[feature]} in the item offered by the user.<br>'
    return diff


def get_contrast_exp(p, q, features):
    diff = ""
    for feature in features:
        if types.is_categorical_dtype(p[feature]):
            diff += f'The recommended item {feature} is {p[feature]} ' \
                    f'and the item offered by the user is {q[feature]}.<br>'
        else:
            diff += f'The recommended item has a {feature} of {p[feature]} ' \
                    f'compared to {q[feature]} in the item offered by the user.<br>'
    return diff
