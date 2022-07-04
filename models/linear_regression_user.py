from sklearn.linear_model import LinearRegression
from explanations.fetch_data import get_items_rated

class linear_regression_user:
    def __init__(self, user, data):
        clean_items, clean_users, ratings_data = data.get_clean_data()
        user_id = user.name
        x_data, y_data = get_items_rated(user_id, ratings_data, clean_items)

        self.linear_regression_model = LinearRegression()
        self.linear_regression_model.fit(x_data, y_data)

    def predict(self, x, *args):
        return self.linear_regression_model.predict(x)
