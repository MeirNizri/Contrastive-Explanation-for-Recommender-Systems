import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

from datasets import cellphones
from models import my_linear_regression_model
import explanations.items_comparison as exp
from explanations.fetch_data import get_user, get_items_rated, get_recommended_item

# get data
data = cellphones()
items_data, users_data, ratings_data = data.get_data()
clean_items, clean_users, _ = data.get_clean_data()

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
def get_data():
    # get random_item_raw or specific user
    user = get_user(clean_users)
    user_id = user.name

    # get data on items the user rated and train Model
    X, y = get_items_rated(user_id, ratings_data, clean_items)
    model = my_linear_regression_model(X, y)

    # get data of items rated by user
    items_rated_raw = items_data.loc[X.index]
    ratings_date = ratings_data.loc[ratings_data['user_id'] == user_id]
    ratings_date.set_index('item_id', inplace=True)
    items_rated_raw = pd.concat([items_rated_raw, ratings_date['date'], y], axis=1)
    items_rated_raw = items_rated_raw.rename(columns={'date': 'Ratings date'})
    items_rated_raw.reset_index(drop=True, inplace=True)
    items_rated_raw_html = items_rated_raw.to_html(classes='w3-table-all w3-centered w3-hoverable w3-striped')

    # get all unrated items and update "time from release" feature
    items_not_rated = clean_items.loc[clean_items.index.drop(X.index)]
    user_last_rate = pd.to_datetime(user['last_rate'])
    release_to_last_rate = lambda x: (user_last_rate - datetime.strptime(str(x), '%Y')).days / 365.0
    items_not_rated['Time from release'] = items_not_rated['Release date'].map(release_to_last_rate)

    # get item with max rating prediction
    recommended_item = get_recommended_item(items_not_rated, model)
    recommended_item_raw = items_data.loc[recommended_item.name]
    recommended_item_raw_html = recommended_item_raw.to_frame().T.to_html(
        classes='w3-table-all w3-centered w3-hoverable w3-striped',
        index_names=False
    )

    # get random_item_raw item
    items_not_rated = items_not_rated.loc[items_not_rated.index.drop(recommended_item.name)]
    random_item = items_not_rated.sample().squeeze()
    random_item_raw = items_data.loc[random_item.name]
    random_item_raw_html = random_item_raw.to_frame().T.to_html(
        classes='w3-table-all w3-centered w3-hoverable w3-striped',
        index_names=False
    )

    # get our explanation
    our_exp_features = exp.contrast_exp(model, user_id, recommended_item, random_item)[:3]
    our_exp = exp.get_contrast_exp(recommended_item_raw, random_item_raw, our_exp_features)
    max_len = min(3, len(our_exp_features))
    # benchmarks explanations
    random_exp_features = exp.random_contrast_exp(model, user_id, recommended_item, random_item)[:max_len]
    random_exp = exp.get_contrast_exp(recommended_item_raw, random_item_raw, random_exp_features)
    lr_exp_features = exp.lr_contrast_exp(X, y)[:max_len]
    lr_exp = exp.get_contrast_exp(recommended_item_raw, random_item_raw, lr_exp_features)
    test_exp = exp.test_contrast_exp(recommended_item_raw, random_item_raw, max_len)
    # combine all explanations to list
    explanations = [our_exp, random_exp, lr_exp, test_exp]

    # return all data needed for server in json format
    response = jsonify({
        'user_id': str(user_id),
        'items_rated': items_rated_raw_html,
        'recommended_item': recommended_item_raw_html,
        'random_item': random_item_raw_html,
        'explanations': explanations
    })
    print(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
