import sys
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse

from explanations.fetch_data import get_items_rated


class mlp_user:
    def __init__(self, user, data):
        clean_items, clean_users, ratings_data = data.get_clean_data()
        user_id = user.name
        x_data, y_data = get_items_rated(user_id, ratings_data, clean_items)

        # set MLP parameters
        layers = [(100, 50, 30, 10, 5), (20, 10, 5, 2), (5, 5, 5), (10)]
        mlp_best_result = sys.float_info.max

        for (j, layer) in enumerate(layers):
            # create a Logistic Regression classifier instance and compute the prediction
            mlp_regressor = MLPRegressor(hidden_layer_sizes=layer,
                                         solver='adam',
                                         learning_rate="adaptive",
                                         max_iter=500)
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
        return self.mlp_best_model.predict(x)
