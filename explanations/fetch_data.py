import pandas as pd


def get_user(users, user_id=None):
    if user_id:  # get specific user
        return users.loc[user_id]
    else:  # get random user
        return users.sample(n=1).squeeze()


def get_items_rated(user_id, ratings_data, items_data):
    # get data on items rated by the user
    items_rated = ratings_data.loc[ratings_data['user_id'] == user_id]
    items_rated.set_index('item_id', inplace=True)
    items_rated_data = items_data.loc[items_rated.index]
    items_rated_data = pd.concat([items_rated_data, items_rated['Time from release']], axis=1)

    # get the ratings of the items rated by the user
    users_ratings = items_rated['rating'].astype('int')

    return items_rated_data, users_ratings


def get_recommended_item(items, model):
    predictions = model.predict(items)
    return items.iloc[predictions.argmax()]
