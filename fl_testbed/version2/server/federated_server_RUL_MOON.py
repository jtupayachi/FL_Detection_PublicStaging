


"""

Main python file for running the federated client, parameter and the Strategy class are initialized.

"""
#NEW LIBRARIES
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    r2_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense
# from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
# from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from autosklearn.regression import AutoSklearnRegressor
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2,L1,L1L2
import tensorflow as tf
#LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation, Dense
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from datetime import datetime
from tensorflow.keras.layers import Dropout,Conv1D,MaxPooling1D,Flatten, Activation
from tensorflow.keras.regularizers import L1L2
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import Sequential
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# from autosklearn.regression import AutoSklearnRegressor
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow.keras.regularizers import L2,L1,L1L2
import pandas as pd
import os
import numpy as np
RNDSEED = np.random.seed(39)
PRECISION = 4 # 3 of digits to keep after the decimal point
from sklearn import tree
from sklearn import metrics
from sklearn.metrics import f1_score, cohen_kappa_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
from keras.regularizers import l2, l1
from tensorflow.keras.layers import Dropout
from keras.layers import LeakyReLU
import json
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import f1_score, cohen_kappa_score
import tensorflow as tf
from sklearn.model_selection import KFold
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import pickle
import random as rn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    r2_score,
    mean_absolute_percentage_error,
    max_error,
    mean_squared_log_error,
    median_absolute_error,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from abc import ABC
import pickle
from typing import Dict, Tuple
from tensorflow.keras.models import Sequential, save_model, load_model
from flwr.common import Config, NDArrays, Scalar
from sklearn.model_selection import KFold
from sklearn import preprocessing
import numpy as np
import pandas as pd
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import os
from sklearn.model_selection import train_test_split
# from autosklearn.regression import AutoSklearnRegressor
from tensorflow.keras.regularizers import L2,L1,L1L2
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
RNDSEED = np.random.seed(39)
PRECISION = 4 # 3 of digits to keep after the decimal point
from sklearn.preprocessing import StandardScaler, MinMaxScaler
RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os
import pickle
import random as rn
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

#AS IS
from ast import Dict
from inspect import Parameter
import flwr as fl
import tensorflow as tf
import os

# Configure GPU memory growth to prevent OOM errors in parallel execution
# Allow GPU selection via environment variable (for multi-GPU systems)
gpu_id = os.environ.get('GPU_ID', None)
if gpu_id is not None:
    # Set specific GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"Using GPU {gpu_id}")

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"GPU memory growth config error: {e}")

# from tensorflow import keras
import sys
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import calendar
import time
import matplotlib.pyplot as plt

# scaling and train test split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# creating a model
# from tensorflow.keras.models import Sequential, save_model, load_model
# from tensorflow.keras.layers import Dense, Activation
# from tensorflow.keras.optimizers import Adam

# evaluation on test data
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
)
from sklearn.metrics import classification_report, confusion_matrix
from typing import Optional, Tuple, Union, List
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn import preprocessing
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
)
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import argparse
import sys

import json

import tensorflow as tf
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import pickle
import random as rn
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.random import set_seed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    explained_variance_score,
    accuracy_score,
    r2_score,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    recall_score,
    cohen_kappa_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold



def load_data(TRANSFORMED_FOLDER,data_file_name,dfn_test_x,dfn_test_y):
        
    print("INI")

    #TRAINING
    with open(
            TRANSFORMED_FOLDER + data_file_name,
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        # self.df = pickle.load(file)
        df=pd.read_pickle(file)
    

    #TEST
    with open(
            TRANSFORMED_FOLDER + dfn_test_x.replace('\n', ''),
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        X_test = pickle.load(file)

    with open(
            TRANSFORMED_FOLDER + dfn_test_y.replace('\n', ''),
            # self.concatenated_identifier +
            # self.FILE_NAME + "y_train.pkl",
            "rb",
    ) as file:
        y_test = pickle.load(file)

    del file



    return df,X_test,y_test

    
    
    
    
    

def model_definition(df,test_inputs,test_out,RNDSEED):

    print("#ASSIGNAMEMNT")
    



    train_inputs=np.array(df.X.apply(lambda x: np.array(x)).tolist())
    # train_inputs=train_inputs
    # self.df.X.apply(lambda row: row[:][:-1])
    train_out=np.array(df.y.apply(lambda x: np.array(x)).tolist())
    # np.array(self.df.y).reshape(-1,1)
    

    # print(train_inputs)
    # print(train_out)

    print(train_inputs.shape)
    print(train_out.shape)
    # input()
    #TESTS PKL
    # JT CHANGES!!!
    # test_inputs=X_test
    # test_out=y_test

    # del X_test
    # del y_test

    


    

    stra_train_inputs=train_inputs[:,:,-1]

    # Use the same function above for the validation set WE JUST SPLIT IT IN 0.25 and 0.75 OF THE PREVIOUS SPLIT
    train_inputs, vals_inputs, train_out, vals_out = train_test_split(train_inputs, train_out, 
        test_size=0.25,shuffle=True, random_state= RNDSEED,stratify=stra_train_inputs) # 0.25 x 0.8 = 0.2





    #DROPPING STATUS

    print(train_inputs.shape)
    train_inputs=train_inputs[:,:,:-1]
    test_inputs=test_inputs[:,:,:-1]
    vals_inputs=vals_inputs[:,:,:-1]
    print(train_inputs.shape)
    # input()

    print("train_out")
    print(train_out.shape)
    # polyline = np.array(np.linspace(0,len(train_out),len(train_out) ) )
    scaler=MinMaxScaler(feature_range=(0, 1))
    train_out=scaler.fit_transform(train_out)
    # train_out = train_out.apply(NormalizeData)
    # NormalizeData(polyline)
    train_out=train_out.reshape(-1,1)
    

    print("test_out")
    print(test_out.shape)
    # polyline = np.array(np.linspace(0,len(test_out),len(test_out) ) )
    scaler=MinMaxScaler(feature_range=(0, 1))
    test_out=scaler.fit_transform(test_out)
    # test_out = test_out.apply(NormalizeData)
    # test_out = NormalizeData(polyline)
    test_out=test_out.reshape(-1,1)


    print("vals_out")
    # print(vals_out)
    print(vals_out.shape)
    # polyline = np.array(np.linspace(0,len(vals_out),len(vals_out) ) )
    scaler=MinMaxScaler(feature_range=(0, 1))
    vals_out=scaler.fit_transform(vals_out)
    # vals_out = vals_out.apply(NormalizeData)
    # vals_out = NormalizeData(polyline)
    vals_out=vals_out.reshape(-1,1)
    # print(vals_out)
    print(vals_out.shape)
    

    #NOW THE SEQUENCES
    "train_inputs"
    for seq in range(train_inputs.shape[0]):
        scaler=StandardScaler()
        
        train_inputs[seq]=scaler.fit_transform(train_inputs[seq])

    "test_inputs"
    for seq in range(test_inputs.shape[0]):
        scaler=StandardScaler()
        
        test_inputs[seq]=scaler.fit_transform(test_inputs[seq])



    "vals_inputs"
    for seq in range(vals_inputs.shape[0]):
        scaler=StandardScaler()
        
        vals_inputs[seq]=scaler.fit_transform(vals_inputs[seq])


    print(train_inputs.shape)
    print(test_inputs.shape)
    print(vals_inputs.shape)

    

    
    tf.keras.backend.clear_session()
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=conf)

    nb_features = train_inputs.shape[2]
    sequence_length  = train_inputs.shape[1]
    nb_out = train_out.shape[1]

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape = (sequence_length, nb_features), return_sequences = True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(nb_out, activation = 'relu')
    ])

    

    model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-7), metrics =['mse','mae'])

    
    
    
    
    
    


    # JT CHANGES
    # X_train=train_inputs
    # y_train=train_out
    # X_vals=vals_inputs
    # y_vals=vals_out
    # X_test=test_inputs
    # y_test=test_out



    return model, train_inputs,train_out,vals_inputs,vals_out,test_inputs,test_out


def modeling(model, train_inputs,train_out,vals_inputs,vals_out,X_test,y_test,batch_size,epochs):

    # JT CHJANGES!!! 
    # train_inputs=X_train
    # train_out=y_train
    # vals_inputs=X_vals
    # vals_out=y_vals
    
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

    # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-7), metrics =['mse','mae'])


    #FAST AI SEE IF TRIANING IMPROVES !

    es = EarlyStopping(monitor="val_loss",
            mode="auto",
            verbose=0,
            patience=1,
            min_delta=0.0001,
            restore_best_weights=True)
    # 1420492
    # history = model.fit(train_inputs, train_out, epochs = 20, callbacks = [lr])
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

    from  datetime import datetime

    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history=model.fit(train_inputs,train_out,validation_data= (vals_inputs,vals_out) ,epochs=epochs ,verbose=0)
    return history


def main() -> None:


        # Arguments
    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "-cm",
        "--clients_max",
        help="Maximun number of clients",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-e",
        "--epochs",
        help="epochs",
        type=int,
        required=True,
        default=0,
    )

    parser.add_argument("-ip", "--ip", help="IP address", type=str)

    parser.add_argument("-dfn_test_x",
                        "--dfn_test_x",
                        help="dfn_test_x",
                        required=True,
                        type=str)
    

    parser.add_argument("-dfn_test_y",
                    "--dfn_test_y",
                    help="dfn_test_y",
                    required=True,
                    type=str)

    parser.add_argument("-dfn",
                        "--dfn",
                        help="dfn",
                        required=True,
                        default=None)

    parser.add_argument("-r",
                        "--rounds",
                        help="federated rounds",
                        required=True,
                        default=None)


    parser.add_argument("-tau",
                        "--tau",
                        help="tau",
                        type=float)

    parser.add_argument("-slr",
                        "--slr",
                        help="slr",
                        type=float)

    parser.add_argument("-mu",
                        "--mu",
                        help="MOON contrastive loss weight",
                        type=float,
                        required=True)

    parser.add_argument("-temperature",
                        "--temperature",
                        help="MOON temperature parameter",
                        type=float,
                        required=True)

    args = parser.parse_args()

    
    clients_max = int(args.clients_max)
    epochs = int(args.epochs)
    ip = str(args.ip)
    dfn_test_y = str(args.dfn_test_y)
    dfn_test_x = str(args.dfn_test_x)
    dfn = str(args.dfn)
    rounds = int(args.rounds)
    tau = float(args.tau)
    slr = float(args.slr)
    mu = float(args.mu)
    temperature = float(args.temperature)

    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)


    # Load and compile model for
    #WE LOAD THE DATA
    TRANSFORMED_FOLDER= "fl_testbed/version2/data/transformed/"
    df,X_test,y_test=load_data(TRANSFORMED_FOLDER,dfn,dfn_test_x,dfn_test_y)
    #WE LOAD THE MODEL UP TO MODEL COMPILE
    #JT CHANGES
    del dfn,dfn_test_x,dfn_test_y
    model, X_train,y_train,X_vals,y_vals,X_test,y_test=model_definition(df,X_test,y_test,RNDSEED)
    #JT CHANGES
    del df

    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation


    #WE CREATE A STRATEGY
    strategy = fl.server.strategy.FedOpt(
        fraction_fit=0.1,
        fraction_evaluate=0.1,
        # eta = 1e-1,
        eta_l = 1e-1,
        beta_1 = 0.0,
        beta_2 = 0.0,
        # tau = 1e-9,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=5, #DEFAULT 10
        evaluate_fn=None,  # DISABLED: Server-side evaluation removed to save GPU memory
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        eta=slr,
        tau=tau,
        )

        # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        # server_address="0.0.0.0:8080",
        # config=fl.server.ServerConfig(num_rounds=4),
        server_address=str(ip),  # ip parameter already includes port (e.g., "172.18.1.10:8080")
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        
    )

def get_evaluate_fn(model,X_test,y_test):
    """Return an evaluation function for server-side evaluation."""

    # # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

    # # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[45000:50000], y_train[45000:50000]
    #WE EMPLOY OUR SEPARATED INDEPENDENT TEST SET

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, mse, mae = model.evaluate(X_test,y_test,verbose=0)
        y_pred = model.predict(X_test[0:],verbose=0) ## using the untinted dataset!
    
        print('R^2:', metrics.r2_score(y_test[0:], y_pred))
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test[0:], y_pred))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(y_test[0:], y_pred))
        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(y_test[0:], y_pred))
        print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(y_test[0:], y_pred))) # np.sqrt

        print('Explained Variance Score:', metrics.explained_variance_score(y_test[0:], y_pred))
        print('Max Error:', metrics.max_error(y_test[0:], y_pred))
        print('Mean Squared Log Error:', metrics.mean_squared_log_error(y_test[0:], y_pred))
        print('Median Absolute Error:', metrics.median_absolute_error(y_test[0:], y_pred))

        return loss, {"mse": mse,"mae":mae,"r2":metrics.r2_score(y_test[0:], y_pred)}


    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16, #JT ORGININAL 32 NOW 256
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10   #JT NOW 100 AND 200 --> 5 if serv <4 else 10
    return {"val_steps": val_steps}
    





if __name__ == "__main__":
    main()









# Create strategy and run server

# strategy = SaveModelStrategy(
#     rndseed=RNDSEED,
#     epochs=epochs,
#     dfn=dfn,
    
    
#     clients_max=clients_max,
#     ip=ip,
#     dfn_test_y=dfn_test_y,
#     dfn_test_x=dfn_test_x,
#     # fq=float(fq),


#     weights=None,
#     min_available_clients=clients_max,
#     min_evaluate_clients=clients_max,
#     min_fit_clients=clients_max,
#     df=None,
# )  # ,evaluate_fn=evaluate

# # Start Flower server for three rounds of federated learning
# fl.server.start_server(
#     server_address=str(ip) + ":8080",
#     config=fl.server.ServerConfig(num_rounds=rounds),
#     grpc_max_message_length=1024 * 1024 * 1024,
#     strategy=strategy,
# )



