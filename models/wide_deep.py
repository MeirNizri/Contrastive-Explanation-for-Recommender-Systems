import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
from tempfile import TemporaryDirectory
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import cellphones

from recommenders.utils.constants import DEFAULT_PREDICTION_COL, SEED
from recommenders.utils import tf_utils
from recommenders.datasets.pandas_df_utils import user_item_pairs
from recommenders.datasets.python_splitters import python_random_split
import recommenders.evaluation.python_evaluation as evaluator
import recommenders.models.wide_deep.wide_deep_utils as wide_deep


class wide_deep_model:
    def __init__(self):
        ITEM_FEAT_COL = ['internal memory','RAM','performance', 'main camera', 'selfie camera', 
                        'battery size', 'screen size', 'weight', 'price', 'release date', 
                        'Apple', 'Asus', 'Samsung', 'Google', 'OnePlus', 'Oppo', 'Vivo' ,'Xiaomi',	
                        'Sony', 'Motorola', 'iOS',	'Android']
        USER_COL = 'user_id'
        ITEM_COL = 'cellphone_id'
        RATING_COL = 'rating'


        # Metrics to use for evaluation
        RATING_METRICS = [
            evaluator.rmse.__name__,
            evaluator.mae.__name__,
        ]
        RANDOM_SEED = SEED  # Set seed for deterministic result

        # Train and test set pickle file paths. If provided, use them. Otherwise, download the MovieLens dataset.
        DATA_DIR = None
        TRAIN_PICKLE_PATH = None
        TEST_PICKLE_PATH = None
        # Model checkpoints directory. If None, use temp-dir.
        MODEL_DIR = None

        #Hyperparameters
        MODEL_TYPE = 'wide_deep'
        STEPS = 5000  # Number of batches to train
        BATCH_SIZE = 32
        # Wide (linear) model hyperparameters
        LINEAR_OPTIMIZER = 'adagrad'
        LINEAR_OPTIMIZER_LR = 0.0621  # Learning rate
        LINEAR_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
        LINEAR_L2_REG = 0.0
        LINEAR_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
        # DNN model hyperparameters
        # DNN_OPTIMIZER = 'adam'
        DNN_OPTIMIZER = 'adadelta'
        DNN_OPTIMIZER_LR = 0.0621
        DNN_L1_REG = 0.0           # Regularization rate for FtrlOptimizer
        DNN_L2_REG = 0.0
        DNN_MOMENTUM = 0.0         # Momentum for MomentumOptimizer or RMSPropOptimizer
        # Layer dimensions. Defined as follows to make this notebook runnable from Hyperparameter tuning services like AzureML Hyperdrive
        DNN_HIDDEN_LAYER_1 = 0     # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_2 = 64    # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_3 = 128   # Set 0 to not use this layer
        DNN_HIDDEN_LAYER_4 = 512   # Note, at least one layer should have nodes.
        DNN_HIDDEN_UNITS = [h for h in [DNN_HIDDEN_LAYER_1, DNN_HIDDEN_LAYER_2, DNN_HIDDEN_LAYER_3, DNN_HIDDEN_LAYER_4] if h > 0]
        DNN_USER_DIM = 16          # User embedding feature dimension
        DNN_ITEM_DIM = 32          # Item embedding feature dimension
        DNN_DROPOUT = 0.8
        DNN_BATCH_NORM = 1         # 1 to use batch normalization, 0 if not.


        if MODEL_DIR is None:
            TMP_DIR = TemporaryDirectory()
            model_dir = TMP_DIR.name
        else:
            if os.path.exists(MODEL_DIR) and os.listdir(MODEL_DIR):
                raise ValueError(
                    "Model exists in {}. Use different directory name or "
                    "remove the existing checkpoint files first".format(MODEL_DIR)
                )
            TMP_DIR = None
            model_dir = MODEL_DIR


        use_preset = (TRAIN_PICKLE_PATH is not None and TEST_PICKLE_PATH is not None)
        if not use_preset:
            cellphones_data = cellphones()
            data = cellphones_data.get_all_data()

        if not use_preset:
            train, test = python_random_split(data, ratio=0.9, seed=RANDOM_SEED)
        else:
            train = pd.read_pickle(path=TRAIN_PICKLE_PATH if DATA_DIR is None else os.path.join(DATA_DIR, TRAIN_PICKLE_PATH))
            test = pd.read_pickle(path=TEST_PICKLE_PATH if DATA_DIR is None else os.path.join(DATA_DIR, TEST_PICKLE_PATH))
            data = pd.concat([train, test])
        print("{} train samples and {} test samples".format(len(train), len(test)))

        # Unique items and users in the dataset
        item_feat_shape = len(ITEM_FEAT_COL)
        items = cellphones_data.get_clean_data()
        users = cellphones_data.get_clean_users()
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
            dnn_batch_norm=(DNN_BATCH_NORM==1),
            log_every_n_iter=max(1, STEPS//10),  # log 10 times
            save_checkpoints_steps=max(1, STEPS//10),
            seed=RANDOM_SEED
        )

        cols = {
            'col_user': USER_COL,
            'col_item': ITEM_COL,
            'col_rating': RATING_COL,
            'col_prediction': DEFAULT_PREDICTION_COL,
        }

        # Prepare ranking evaluation set, i.e. get the cross join of all user-item pairs
        ranking_pool = user_item_pairs(
            user_df=users,
            item_df=items,
            user_col=USER_COL,
            item_col=ITEM_COL,
            user_item_filter_df=train,  # Remove seen items
            shuffle=True,
            seed=RANDOM_SEED
        )

        # Define training input (sample feeding) function
        train_fn = tf_utils.pandas_input_fn(
            df=train,
            y_col=RATING_COL,
            batch_size=BATCH_SIZE,
            num_epochs=None,  # We use steps=TRAIN_STEPS instead.
            shuffle=True,
            seed=RANDOM_SEED,
        )

        print(
            "Training steps = {}, Batch size = {} (num epochs = {})"
            .format(STEPS, BATCH_SIZE, (STEPS*BATCH_SIZE)//len(train))
        )
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

        try:
            self.wide_deep_model.train(
                input_fn=train_fn,
                steps=STEPS
            )
        except tf.train.NanLossDuringTrainingError:
            import warnings
            warnings.warn(
                "Training stopped with NanLossDuringTrainingError. "
                "Try other optimizers, smaller batch size and/or smaller learning rate."
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
        # convert dtypes
        x = x.convert_dtypes()
        x.iloc[:, 11:23] = x.iloc[:, 11:23].astype('bool')
        x.iloc[:, [0,1,2,4,5,6,8,9,10,23,24,25,26]] = x.iloc[:, [0,1,2,4,5,6,8,9,10,23,24,25,26]].astype('int64')
        x.iloc[:, [3,7]] = x.iloc[:, [3,7]].astype('float64')
        
        # get prediction
        prediction_generator = list(self.wide_deep_model.predict(input_fn=tf_utils.pandas_input_fn(df=x)))
        predictions = [p['predictions'][0] for p in prediction_generator]
        return np.asarray(predictions)