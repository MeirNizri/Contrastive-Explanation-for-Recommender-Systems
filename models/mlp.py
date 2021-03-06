import sys
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

class mlp:
    def __init__(self,  user, data):
        self.user = user
        self.user['last_rate'] = pd.to_datetime(self.user['last_rate']).year

        all_data = data.get_all_data()
        all_data['last_rate'] = pd.to_datetime(all_data['last_rate']).dt.year
        y_data = all_data['rating']
        x_data = all_data.drop(columns=['rating'])

        # set MLP parameters
        layers = [(100, 50, 30, 10, 5), (20, 10, 5, 2), (5, 5, 5), (10)]
        mlp_best_result = sys.float_info.max

        for (j, layer) in enumerate(layers):
            # create a Logistic Regression classifier instance and compute the prediction
            mlp_regressor = MLPRegressor(hidden_layer_sizes=layer,
                                         solver='adam',
                                         learning_rate="adaptive",
                                         max_iter=1000)
            mlp_regressor.fit(x_data, y_data)
            y_pred = mlp_regressor.predict(x_data)

            # calculate F-measures and save best model
            error = mse(y_pred, y_data)
            if error < mlp_best_result:
                mlp_best_result = error
                self.mlp_best_model = mlp_regressor
        # print best model parameters
        params = self.mlp_best_model.get_params()
        print('best result for: optimizer = %s, layer = %s, mse = %f' %
              (params['solver'], params['hidden_layer_sizes'], mlp_best_result))

    def predict(self, x, *args):
        user_x = pd.DataFrame([self.user] * len(x)).reset_index()
        items_x = x.reset_index(drop=True)
        user_items = pd.concat([user_x, items_x], axis=1)
        return self.mlp_best_model.predict(user_items)
