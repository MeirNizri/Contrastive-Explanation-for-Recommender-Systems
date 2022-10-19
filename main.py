from flask import Flask, jsonify, render_template
from flask_cors import CORS
from datetime import datetime
import gspread
import pandas as pd

from datasets import cellphones
from models import mlp_model, wide_deep_model
from explanations import data_util
import explanations.items_comparison as exp

# get data
data = cellphones()
cellphones_data = data.get_data()
clean_data = data.get_clean_data()
all_data = data.get_all_data()
# create model
wide_deep = wide_deep_model()
# create flask app
app = Flask(__name__)
CORS(app)


@app.route("/")
def get_items_to_rate():
    # Sample 10 cellphones
    phones_to_rate = cellphones_data.sample(10)
    phones_id = list(phones_to_rate.index)

    # convert phone data to html code and send in json format
    phones_to_rate_html = data_util.rate_to_html(phones_to_rate)
    response = jsonify({
        'phones_to_rate': phones_to_rate_html,
        'phones_id': phones_id
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/survey/")
def survey():
    return render_template('survey.html')

@app.route("/<model>/<phones_id>/<ratings>/<birth_year>/<gender>/")
def get_comparison_data(model, phones_id, ratings, birth_year, gender):
    # get clean data of the cellphones rated
    phones_id = phones_id.split(",")
    phones_id = list(map(int, phones_id))
    phones_data = clean_data.loc[phones_id]
    # convert ratings to array of integers
    ratings = ratings.split(",")
    ratings = list(map(int, ratings))

    # train Model
    if model == "mlp":
        model = mlp_model(phones_data, ratings, all_data)
    else:
        model = wide_deep

    # get cellphone with max rating prediction
    phones_not_rated = clean_data.loc[clean_data.index.drop(phones_data.index)]
    phones_not_rated['user_id'] = -1
    phones_not_rated['age'] = 2022 - int(birth_year)
    phones_not_rated['gender'] = 0 if gender == "Female" else 1
    phones_not_rated['occupation'] = 0
    predictions = model.predict(phones_not_rated)
    recommended_item = phones_not_rated.iloc[predictions.argmax()]
    recommended_item_raw = cellphones_data.loc[recommended_item.name]
    recommended_item_html = data_util.phone_to_html(recommended_item_raw)

    # get random cellphone
    phones_not_rated.drop([recommended_item.name], inplace=True)
    random_item = phones_not_rated.sample().squeeze()
    random_item_raw = cellphones_data.loc[random_item.name]
    random_item_html = data_util.phone_to_html(random_item_raw)

    # get our explanations
    our_exp_features = exp.contrast_exp(model, recommended_item, random_item, clean_data)
    our_exp = data.get_contrast_exp(recommended_item_raw, random_item_raw, our_exp_features)
    max_len = min(3, len(our_exp_features))
    # benchmarks explanations
    random_exp_features = exp.random_contrast_exp(recommended_item, random_item)
    random_exp = data.get_contrast_exp(recommended_item_raw, random_item_raw, random_exp_features, max_len)
    lr_exp_features = exp.lr_contrast_exp(phones_data, ratings)
    lr_exp = data.get_contrast_exp(recommended_item_raw, random_item_raw, lr_exp_features, max_len)
    test_exp = exp.test_contrast_exp()

    # send data in json format
    response = jsonify({
        'recommended_item': recommended_item_html,
        'random_item': random_item_html,
        'explanations': [our_exp, random_exp, lr_exp, test_exp]
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route("/<age>/<gender>/<occupation>/<phones_id>/<ratings>/<explanations>/<recommended_phone>/<random_phone>/<our_exp>/")
def add_new_user(age, gender, occupation, phones_id, ratings, explanations, recommended_phone, random_phone, our_exp):
    # get clean data of the cellphones rated
    phones_id = phones_id.split(",")
    phones_id = list(map(int, phones_id))
    # convert ratings to array of integers
    ratings = ratings.split(",")
    ratings = list(map(int, ratings))
    explanations = explanations.split(",")

    # connect to google sheet
    account = gspread.service_account(filename='datasets/contrast-explanation-rs-5731512d8ac1.json')
    spreadsheet = account.open("cellphones ratings")
    ratings_sheet = spreadsheet.worksheet("survey_ratings")
    users_sheet = spreadsheet.worksheet("survey_users")

    # add rating data to the sheet
    rating_data = pd.DataFrame(ratings_sheet.get_all_records())
    if not rating_data.empty:
        new_user_id = rating_data.iloc[-1]['user id'] + 1
    else:
        new_user_id = 0
    new_rating_data = pd.DataFrame({
        'user id': [new_user_id] * len(ratings),
        'cellphone id': phones_id,
        'rating': ratings,
        'explanations': explanations,
        'datetime': str(datetime.now())})
    rating_data = pd.concat([rating_data, new_rating_data])
    ratings_sheet.update([rating_data.columns.values.tolist()] + rating_data.values.tolist())

    # add new user data to the sheet
    users_data = pd.DataFrame(users_sheet.get_all_records())
    recommended_phone = recommended_phone.replace("%20", " ")
    random_phone = random_phone.replace("%20", " ")
    our_exp = our_exp.replace("%20", " ")
    new_user_data = pd.DataFrame({
        'user id': [new_user_id],
        'age': [2022 - int(age)],
        'gender': [gender],
        'occupation': [occupation],
        'recommended phone': [recommended_phone],
        'random phone': [random_phone],
        'our exp': [our_exp]})
    users_data = pd.concat([users_data, new_user_data])
    users_sheet.update([users_data.columns.values.tolist()] + users_data.values.tolist())

    # return all data needed for server in json format
    response = jsonify({'empty': " "})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
