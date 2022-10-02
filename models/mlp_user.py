from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error as mse
import sys


class mlp_user:
    def __init__(self, x, y):

        # set MLP parameters
        layers = [(100, 50, 30, 10, 5), (20, 10, 5, 2), (50, 50, 50, 50), (100,)]
        mlp_best_result = sys.float_info.max

        for layer in layers:
            # create a Logistic Regression classifier instance and compute the prediction
            mlp_regressor = MLPRegressor(hidden_layer_sizes=layer,
                                         learning_rate_init=0.01,
                                         max_iter=5000)
            mlp_regressor.fit(x, y)
            y_pred = mlp_regressor.predict(x)

            # calculate loss and save best model
            error = mse(y, y_pred)
            if error < mlp_best_result:
                mlp_best_result = error
                self.mlp_best_model = mlp_regressor
        # print best model parameters
        params = self.mlp_best_model.get_params()
        print('best result for: optimizer = %s, layer = %s, mse = %f' %
              (params['solver'], params['hidden_layer_sizes'], mlp_best_result))

    def predict(self, x):
        return self.mlp_best_model.predict(x)
