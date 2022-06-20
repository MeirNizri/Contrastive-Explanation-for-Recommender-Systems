from sklearn.linear_model import LinearRegression

class my_linear_regression_model:
    def __init__(self, x_data, y_data):
        self.linear_regression_model = LinearRegression()
        self.linear_regression_model.fit(x_data, y_data)

    def predict(self, x, *args):
        return self.linear_regression_model.predict(x)
