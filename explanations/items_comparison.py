import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from .data_util import sample_between, get_diff_features, get_diff_indexes

irrelevant_features = ['cellphone_id', 'user_id',' age', 'gender', 'occupation']


def contrast_exp(rs_model, p, q, data):
    # get p and q ratings from the recommendation model, p should be better than q
    p_copy, q_copy = (p.copy(), q.copy())
    items = pd.concat([p_copy, q_copy], axis=1).T
    p_rating, q_rating = rs_model.predict(items)
    if q_rating > p_rating:
        temp = (p_copy, p_rating)
        p_copy, p_rating = (q_copy, q_rating)
        q_copy, q_rating = temp

    # sample uniformly objects from the items' data that are "between" p and q
    num_samples = 500
    x_data = sample_between(p_copy, q_copy, data, num_samples)
    # x_data.drop_duplicates(inplace=True)
    x_data.reset_index(drop=True, inplace=True)
    y_data = pd.DataFrame(rs_model.predict(x_data), columns=['prediction'])
    # clear samples with all zeros
    null_index = y_data.loc[y_data['prediction'].isnull()].index.tolist()
    x_data = x_data.drop(null_index)
    y_data = y_data.drop(null_index)

    # fit linear regression model and get weights for every feature
    linear_regression_model = LinearRegression().fit(x_data, y_data)
    weights = linear_regression_model.coef_[0]
    features_ = np.array(p.index)
    features_weights_ = dict(zip(features_, weights))
    features_weights_ = sorted(features_weights_.items(), key=lambda x: x[1], reverse=True)
    # print(f'features weights {features_weights}')
    features_weights_ = dict(features_weights_)

    # get the scaled weights by w*(p-q) and save only features with a difference between p and q
    sub_pq = np.subtract(np.multiply(p_copy.to_numpy(), 1), np.multiply(q_copy.to_numpy(), 1))
    scaled_weights = np.multiply(weights, sub_pq)
    diff_idx = get_diff_indexes(p_copy, q_copy)
    scaled_weights = scaled_weights[diff_idx]
    features = np.array(p.index)[diff_idx]

    # integrate between features and weights and sort from high to low
    features_weights = dict(zip(features, scaled_weights))
    features_weights = sorted(features_weights.items(), key=lambda x: x[1], reverse=True)
    # print(f'features weights {features_weights}')
    features_weights = dict(features_weights)
    # print(features_weights)
    
    for k in irrelevant_features:
        features_weights.pop(k, None)

    # build contrastive explanation by iterating the features in order from high weight to low
    explanation = []
    for feature in features_weights.keys():
        explanation.append(feature)
        # Update one feature of q to be as in p
        q_copy[feature] = p_copy[feature]
        items = pd.concat([p_copy, q_copy], axis=1).T
        p_rating, q_rating = rs_model.predict(items)
        
        # check if the rating of q is higher than p
        if q_rating>=p_rating or features_weights[feature]<=0:
            break
        
    return explanation


def random_contrast_exp(p, q):
    # randomly shuffle all the different features between p and q
    diff_features = get_diff_features(p, q)
    diff_features = [f for f in diff_features if f not in irrelevant_features]
    random.shuffle(diff_features)

    return diff_features


def lr_contrast_exp(user_data, user_ratings):
    # normalize data
    x_data = user_data.values
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_data)
    normalize_user_data = pd.DataFrame(x_scaled)

    # fit linear regression model and get weights for every feature
    linear_regression_model = LinearRegression().fit(normalize_user_data, user_ratings)
    weights = linear_regression_model.coef_

    # integrate between features and weights and sort from high to low
    features = np.array(user_data.columns)
    features_weights = dict(zip(features, weights))
    features_weights = sorted(features_weights.items(), key=lambda x: x[1], reverse=True)
    sorted_features = list(map(lambda x: x[0], features_weights))
    sorted_features = [f for f in sorted_features if f not in irrelevant_features]

    # Return features sorted from highest weight to lowest
    return sorted_features


def test_contrast_exp():
    return f'- The recommended cellphone operating system is Ubuntu, ' \
        f'compared to MS-DOS, which is the operating system of the cellphone that Alex preferred.<br>'
