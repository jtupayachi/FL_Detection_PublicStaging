"""

Main python file for running the federated client, parameter and the Strategy class are initialized.

"""
#NEW LIBRARIES
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
# from tensorflow.random import set_seed
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
# import matplotlib.pyplot as plt
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
# from tensorflow import keras
import sys
# import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
import calendar
import time
# import matplotlib.pyplot as plt

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
# import matplotlib.pyplot as plt
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

# import CustomStrategy_OFFSET


# class SaveModelStrategy(CustomStrategy.CustomFedAvg):

    # def aggregate_fit(
    #     self,
    #     server_round: int,
    #     results: List[Tuple[fl.server.client_proxy.ClientProxy,
    #                         fl.common.FitRes]],
    #     failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    # ) -> Tuple[Optional[Parameter], Dict[str, Scalar]]:

    #     # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
    #     aggregated_parameters, aggregated_metrics = super().aggregate_fit(
    #         server_round, results, failures)

    #     if aggregated_parameters is not None:
    #         # Convert `Parameters` to `List[np.ndarray]`
    #         aggregated_ndarrays: List[
    #             np.ndarray] = fl.common.parameters_to_ndarrays(
    #                 aggregated_parameters)

    #         self.load_data()

    #         self.modeling()

    #         self.model.set_weights(aggregated_ndarrays)

    #         self.testing()

    #         # model.evaluate(X_test, clf.transform(y_test))

    #     return aggregated_parameters, aggregated_metrics




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

    return df,X_test,y_test

    
    
    
    
    

def model_definition(df,X_test,y_test,RNDSEED):

    print("#ASSIGNAMEMNT")
    

    # X_train=df.X.values
    # y_train=df.y.values
    X_train=np.array(df.X.apply(lambda x: np.array(x)).tolist())#
    # train_inputs=train_inputs
    # self.df.X.apply(lambda row: row[:][:-1])
    y_train=np.array(df.y.apply(lambda x: np.array(x)).tolist())#.


    # print(X_train)
    # print(y_train)

    print(X_train.shape)
    print(y_train.shape)
    

    # Use the same function above for the validation set
    X_train, X_vals, y_train, y_vals = train_test_split(X_train, y_train, 
        test_size=0.25, random_state= RNDSEED,shuffle=True,stratify=y_train) # 0.25 x 0.8 = 0.2


    from sklearn.preprocessing import StandardScaler,LabelBinarizer
    lbz = LabelBinarizer()

    print("MODELSLSSDLLDSLSD")
    print(X_train.shape)
    print(X_train.dtype)
    # print(X_train)

    scaler=StandardScaler()
    X_train=scaler.fit_transform(X_train)


    y_train=lbz.fit_transform(y_train)



    scaler=StandardScaler()
    X_vals=scaler.fit_transform(X_vals)
    y_vals=lbz.fit_transform(y_vals)


    scaler=StandardScaler()
    X_test=scaler.fit_transform(X_test)
    y_test=lbz.fit_transform(y_test)


    print(X_train.shape)
    print(X_vals.shape)
    print(X_test.shape)


    from  datetime import datetime
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    print(log_dir)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))


    #TRAINING
    tf.keras.backend.clear_session()
    conf = tf.compat.v1.ConfigProto()
    conf.gpu_options.allow_growth=True
    session = tf.compat.v1.Session(config=conf)

    # self.model
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(60, activation="relu", input_dim=X_train.shape[1],kernel_regularizer=L1L2(l2=0.001,l1=0.001)), #Better

    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(30, activation="relu",kernel_regularizer=L2(l2=0.001)),  #Better
    tf.keras.layers.Dropout(0.5),

    # model.add(Dense(40, activation="relu"))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation="relu",kernel_regularizer=L2(l2=0.001)))  #Better
    # output layer
    tf.keras.layers.Dense(5, activation="softmax")])
                # softmax for probability, #values are sigmoid

    # Configure the model and start training
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    
    X_train=np.array(X_train.astype('float64'))
    y_train=np.array(y_train.astype('float64'))
    
    X_vals=np.array(X_vals.astype('float64'))
    y_vals=np.array(y_vals.astype('float64'))
    
    X_test=np.array(X_test.astype('float64'))
    y_test=np.array(y_test.astype('float64'))
    


    return model, X_train,y_train,X_vals,y_vals,X_test,y_test


def modeling(model, X_train,y_train,X_vals,y_vals,X_test,y_test,batch_size,epochs):
    
    lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

    # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-7), metrics =['mse','mae'])


    #FAST AI SEE IF TRIANING IMPROVES !

    es = EarlyStopping(monitor="val_loss",
            mode="auto",
            verbose=2,
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

    history=model.fit(X_train,y_train,validation_data= (X_vals,y_vals) ,epochs=epochs ,verbose=2)
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


    parser.add_argument("-momentum",
                        "--momentum",
                        help="momentum",
                        type=float)

    parser.add_argument("-slr",
                        "--slr",
                        help="slr",
                        type=float)

    args = parser.parse_args()

    
    clients_max = int(args.clients_max)
    epochs = int(args.epochs)
    ip = str(args.ip)
    dfn_test_y = str(args.dfn_test_y)
    dfn_test_x = str(args.dfn_test_x)
    dfn = str(args.dfn)
    rounds = int(args.rounds)
    
    momentum = float(args.momentum)
    slr = float(args.slr)

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
    model, X_train,y_train,X_vals,y_vals,X_test,y_test=model_definition(df,X_test,y_test,RNDSEED)

    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation


    #WE CREATE A STRATEGY
    strategy=fl.server.strategy.FedAvgM(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=2,
            min_evaluate_clients=2,
            min_available_clients=5,
            evaluate_fn=get_evaluate_fn(model,X_test,y_test),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        server_learning_rate=slr,
        server_momentum=momentum,
        )

        # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        # server_address="0.0.0.0:8080",
        # config=fl.server.ServerConfig(num_rounds=4),
        server_address=str(ip) + ":8080",
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
        loss, accuracy = model.evaluate(X_test,y_test,verbose=0)

        # #TODO IMPLEMENT:
        # y_test=y_test #JT
        y_prob = model.predict(X_test,verbose=2) #JT
        y_prob_am=np.argmax(y_prob, axis=1)
        y_test_am=np.argmax(y_test, axis=1)
        macro_roc_auc_ovo = roc_auc_score(y_test, y_prob, multi_class="ovo", average="macro")
        weighted_roc_auc_ovo = roc_auc_score(
            y_test, y_prob, multi_class="ovo", average="weighted"
        )
        macro_roc_auc_ovr = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        weighted_roc_auc_ovr = roc_auc_score(
            y_test, y_prob, multi_class="ovr", average="weighted"
        )
        print(
            "One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(ovo weighted by prevalence)".format(macro_roc_auc_ovo, weighted_roc_auc_ovo)
        )
        print(
            "One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
            "(ovr weighted by prevalence)".format(macro_roc_auc_ovr, weighted_roc_auc_ovr)
        )
        print(classification_report(y_test_am,y_prob_am ))
        print("MCS", matthews_corrcoef(y_test_am,y_prob_am ))

        return loss, {"accuracy": accuracy,"MCS":matthews_corrcoef(y_test_am,y_prob_am ),"ROC_AUC_WEIGHTED":weighted_roc_auc_ovr,"ROC_AUC_MACRO":macro_roc_auc_ovr}

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
    val_steps = 5 if server_round < 4 else 10   #NOW 100 AND 200 --> 5 if serv <4 else 10
    return {"val_steps": val_steps}
    





if __name__ == "__main__":
    main()






