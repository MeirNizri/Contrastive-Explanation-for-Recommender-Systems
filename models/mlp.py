from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
import sys


class mlp_model:
    def __init__(self, user_phones, user_ratings, all_data):
        # set MLP layers
        layers = [(100, 50, 30, 10, 5), (20, 10, 5, 2), (50, 50, 50, 50), (100,)]
        
        # train mlp on user ratings only
        mlp_best_result = sys.float_info.max
        for layer in layers:
            # create a Logistic Regression classifier instance
            mlp_user = MLPRegressor(hidden_layer_sizes=layer,
                                         learning_rate_init=0.01,
                                         max_iter=5000)
            mlp_user.fit(user_phones, user_ratings)
            ratings_pred = mlp_user.predict(user_phones)

            # calculate loss and save best model
            user_error = mse(user_ratings, ratings_pred)
            if user_error < mlp_best_result:
                mlp_best_result = user_error
                self.mlp_user_best = mlp_user
        
        # print best model parameters
        params = self.mlp_user_best.get_params()
        print('best result for user model: optimizer = %s, layer = %s, mse = %f' %
              (params['solver'], params['hidden_layer_sizes'], mlp_best_result))
        
        
        # split data to train and test
        y = all_data['rating']
        X = all_data.drop(columns=['rating', 'cellphone_id'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # train mlp on all users data
        mlp_best_result = sys.float_info.max
        for layer in layers:
            # create a Logistic Regression classifier instance
            mlp_all_users = MLPRegressor(hidden_layer_sizes=layer,
                                         learning_rate_init=0.01,
                                         max_iter=5000)
            mlp_all_users.fit(X_train, y_train)
            y_pred = mlp_all_users.predict(X_test)

            # calculate loss and save best model
            user_error = mse(y_test, y_pred)
            if user_error < mlp_best_result:
                mlp_best_result = user_error
                self.mlp_all_best = mlp_all_users
            
        # print best model parameters
        params = self.mlp_all_best.get_params()
        print('best result for all users: optimizer = %s, layer = %s, mse = %f' %
              (params['solver'], params['hidden_layer_sizes'], mlp_best_result))


    def predict(self, x):
        pred1 = self.mlp_all_best.predict(x)
        x_ = x.drop(columns=['age','gender','user_id'])
        pred2 = self.mlp_user_best.predict(x_)
        return (pred1+pred2)/2
