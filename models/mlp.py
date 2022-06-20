import sys
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error as mse

class my_nn_model:
    def __init__(self, x_data, y_data):
        # set MLP parameters
        layers = [(100, 50, 30, 10, 5), (20, 10, 5, 2), (5, 5, 5), (10)]
        optimizers = ['adam', 'sgd']
        mlp_best_result = sys.float_info.max

        for (i, opt) in enumerate(optimizers):
            for (j, layer) in enumerate(layers):
                # create a Logistic Regression classifier instance and compute the prediction
                mlp_classifier = MLPClassifier(hidden_layer_sizes=layer,
                                               solver=opt,
                                               learning_rate='adaptive',
                                               max_iter=5000)
                mlp_classifier.fit(x_data, y_data)
                y_pred = mlp_classifier.predict(x_data)

                # calculate F-measures and save best model
                error = mse(y_pred, y_data)
                if error < mlp_best_result:
                    mlp_best_result = error
                    self.mlp_best_model = mlp_classifier
        # print best model parameters
        params = self.mlp_best_model.get_params()
        print('best result for: optimizer = %s, layer = %s, mse = %f' %
              (params['solver'], params['hidden_layer_sizes'], mlp_best_result))

    def predict(self, x, *args):
        return self.mlp_best_model.predict(x)
