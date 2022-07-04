from sklearn.linear_model import LinearRegression
import pandas as pd

class linear_regression:
    def __init__(self, user, data):
        self.user = user
        self.user['last_rate'] = pd.to_datetime(self.user['last_rate']).year

        all_data = data.get_all_data()
        all_data['last_rate'] = pd.to_datetime(all_data['last_rate']).dt.year
        y_data = all_data['rating']
        x_data = all_data.drop(columns=['rating'])

        self.linear_regression_model = LinearRegression()
        self.linear_regression_model.fit(x_data, y_data)

    def predict(self, x, *args):
        user_x = pd.DataFrame([self.user] * len(x)).reset_index()
        items_x = x.reset_index(drop=True)
        user_items = pd.concat([user_x, items_x], axis=1)
        return self.linear_regression_model.predict(user_items)
