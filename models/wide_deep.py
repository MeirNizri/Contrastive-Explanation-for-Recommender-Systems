import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tempfile import TemporaryDirectory
import numpy as np
import tensorflow as tf

from datasets import cellphones

from recommenders.utils.constants import DEFAULT_PREDICTION_COL, SEED
from recommenders.utils import tf_utils
from recommenders.datasets.python_splitters import python_random_split
import recommenders.evaluation.python_evaluation as evaluator
import recommenders.models.wide_deep.wide_deep_utils as wide_deep


class wide_deep_model:
    def __init__(self):
        ITEM_FEAT_COL = ['internal memory', 'RAM', 'performance', 'main camera', 'selfie camera',
                         'battery size', 'screen size', 'weight', 'price', 'release date',
                         'Apple', 'Asus', 'Samsung', 'Google', 'OnePlus', 'Oppo', 'Vivo', 'Xiaomi',
                         'Sony', 'Motorola', 'iOS', 'Android']
        USER_COL = 'user_id'
        ITEM_COL = 'cellphone_id'
        RATING_COL = 'rating'

        # Metrics to use for evaluation
        RATING_METRICS = [
            evaluator.rmse.__name__,
            evaluator.mae.__name__,
        ]
        RANDOM_SEED = SEED  # Set seed for deterministic result

        # Hyperparameters
        MODEL_TYPE = 'wide_deep'
        STEPS = 10000  # Number of batches to train
        BATCH_SIZE = 32
        # Wide (linear) model hyperparameters
        LINEAR_OPTIMIZER = 'adagrad'
        LINEAR_OPTIMIZER_LR = 0.3  # Learning rate
        LINEAR_L1_REG = 0.0  # Regularization rate for Ftrl Optimizer
        LINEAR_L2_REG = 0.0
        LINEAR_MOMENTUM = 0.0  # Momentum for MomentumOptimizer or RMSPropOptimizer
        # DNN model hyperparameters
        # DNN_OPTIMIZER = 'adam'
        DNN_OPTIMIZER = 'adadelta'
        DNN_OPTIMIZER_LR = 0.3
        DNN_L1_REG = 0.0  # Regularization rate for FtrlOptimizer
        DNN_L2_REG = 0.0
        DNN_MOMENTUM = 0.0  # Momentum for MomentumOptimizer or RMSPropOptimizer
        # Layer dimensions. Defined as follows to make this notebook runnable from Hyperparameter tuning services like AzureML Hyperdrive
        DNN_HIDDEN_LAYER_1 = 0  # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_2 = 64  # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_3 = 128  # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_4 = 512  # Note, at least one layer should have nodes.
        DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4]
                            if h > 0]
        DNN_USER_DIM = 16  # User embedding feature dimension
        DNN_ITEM_DIM = 64  # Item embedding feature dimension
        DNN_DROPOUT = 0.8
        DNN_BATCH_NORM = 1  # 1 to use batch normalization, 0 if not.

        TMP_DIR = TemporaryDirectory()
        model_dir = TMP_DIR.name

        cellphones_data = cellphones()
        data = cellphones_data.get_all_data()
        train, test = python_random_split(data, ratio=0.9, seed=RANDOM_SEED)
        print("{} train samples and {} test samples".format(len(train), len(test)))

        # Unique items and users in the dataset
        item_feat_shape = len(ITEM_FEAT_COL)
        items = cellphones_data.get_clean_data()
        items.reset_index(inplace=True)
        users = cellphones_data.get_clean_users()
        users.reset_index(inplace=True)
        print("Total {} items and {} users in the dataset".format(len(items), len(users)))

        # Define wide (linear) and deep (dnn) features
        wide_columns, deep_columns = wide_deep.build_feature_columns(
            users=users[USER_COL].values,
            items=items[ITEM_COL].values,
            user_col=USER_COL,
            item_col=ITEM_COL,
            # item_feat_col=ITEM_FEAT_COL,
            crossed_feat_dim=1000,
            user_dim=DNN_USER_DIM,
            item_dim=DNN_ITEM_DIM,
            item_feat_shape=item_feat_shape,
            model_type=MODEL_TYPE
        )

        # Build a model based on the parameters
        self.wide_deep_model = wide_deep.build_model(
            model_dir=model_dir,
            wide_columns=wide_columns,
            deep_columns=deep_columns,
            linear_optimizer=tf_utils.build_optimizer(LINEAR_OPTIMIZER, LINEAR_OPTIMIZER_LR, **{
                'l1_regularization_strength': LINEAR_L1_REG,
                'l2_regularization_strength': LINEAR_L2_REG,
                'momentum': LINEAR_MOMENTUM,
            }),
            dnn_optimizer=tf_utils.build_optimizer(DNN_OPTIMIZER, DNN_OPTIMIZER_LR, **{
                'l1_regularization_strength': DNN_L1_REG,
                'l2_regularization_strength': DNN_L2_REG,
                'momentum': DNN_MOMENTUM,
            }),
            dnn_hidden_units=DNN_HIDDEN_UNITS,
            dnn_dropout=DNN_DROPOUT,
            dnn_batch_norm=(DNN_BATCH_NORM == 1),
            log_every_n_iter=max(1, STEPS // 10),  # log 10 times
            save_checkpoints_steps=max(1, STEPS // 10),
            seed=RANDOM_SEED
        )

        cols = {
            'col_user': USER_COL,
            'col_item': ITEM_COL,
            'col_rating': RATING_COL,
            'col_prediction': DEFAULT_PREDICTION_COL,
        }

        # Define training input (sample feeding) function
        train_fn = tf_utils.pandas_input_fn(
            df=train,
            y_col=RATING_COL,
            batch_size=BATCH_SIZE,
            num_epochs=None,  # We use steps=TRAIN_STEPS instead.
            shuffle=True,
            seed=RANDOM_SEED,
        )

        print(f"Training steps = {STEPS}, Batch size = {BATCH_SIZE} (num epochs = {(STEPS*BATCH_SIZE) // len(train)})")
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        self.wide_deep_model.train(
            input_fn=train_fn,
            steps=STEPS
        )

        # Item rating prediction
        if len(RATING_METRICS) > 0:
            predictions = list(self.wide_deep_model.predict(input_fn=tf_utils.pandas_input_fn(df=test)))
            prediction_df = test.drop(RATING_COL, axis=1)
            prediction_df[DEFAULT_PREDICTION_COL] = [p['predictions'][0] for p in predictions]

            rating_results = {}
            for m in RATING_METRICS:
                result = evaluator.metrics[m](test, prediction_df, **cols)
                rating_results[m] = result
            print(rating_results)


    def predict(self, x):
        s = x.copy()
        if 'cellphone_id' not in s.columns:
            s.reset_index(inplace=True)
            s.rename(columns={"index": "cellphone_id"}, inplace=True)
        # convert dtypes
        s = s.astype('int64')
        s.iloc[:, 11:23] = s.iloc[:, 11:23].astype('bool')

        # get prediction
        prediction_generator = list(self.wide_deep_model.predict(input_fn=tf_utils.pandas_input_fn(df=s)))
        predictions = [p['predictions'][0] for p in prediction_generator]
        return np.asarray(predictions)
