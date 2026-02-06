"""

It entails the FedAvg algorithm and as well as the predefined set of models and its functions.
This allow to perform a testing process on every fitted round and see how the metrics performance improves throughout each of the 
rounds. The FedAvg is based on: Federated Averaging (FedAvg) [McMahan et al., 2016] strategy. Paper: https://arxiv.org/abs/1602.05629

"""
from tensorflow.keras.layers import Dropout
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from keras.layers import LeakyReLU
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
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from sklearn.model_selection import KFold
from aggregate import aggregate, weighted_loss_avg
from strategy import Strategy
from sklearn.preprocessing import StandardScaler
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

#MY IMPORTS

from ast import Dict

from inspect import Parameter
import flwr as fl
import tensorflow as tf
from tensorflow import keras
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
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
# evaluation on test data
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, cohen_kappa_score, matthews_corrcoef, recall_score
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
from flwr.common import Code, EvaluateIns, EvaluateRes, FitIns, FitRes, GetParametersIns, GetParametersRes, Status
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





from abc import ABC
import pickle
from typing import Dict, Tuple
from tensorflow.keras.layers import Dropout
from flwr.common import Config, NDArrays, Scalar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn import preprocessing
from keras.layers import LeakyReLU
import tensorflow as tf
import numpy as np
import pandas as pd
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


#MLP
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

# Libraries
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
from tensorflow.random import set_seed
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





WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# flake8: noqa: E501
class CustomFedAvg(Strategy):
    """Configurable FedAvg strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,line-too-long
    def __init__(
        self,
        *,
        rndseed,
        epochs,
        
        
        clients_max,
        ip,
        dfn_test_y,
        dfn_test_x,
        dfn,
        # fq,
        df=None,


        weights,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable[[int, NDArrays, Dict[
            str, Scalar]], Optional[Tuple[float, Dict[str,
                                                      Scalar]]], ]] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str,
                                                             Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. In case `min_fit_clients`
            is larger than `fraction_fit * available_clients`, `min_fit_clients`
            will still be sampled. Defaults to 1.0.
        fraction_evaluate : float, optional
            Fraction of clients used during validation. In case `min_evaluate_clients`
            is larger than `fraction_evaluate * available_clients`, `min_evaluate_clients`
            will still be sampled. Defaults to 1.0.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_evaluate_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        

        super().__init__()

        if (min_fit_clients > min_available_clients
                or min_evaluate_clients > min_available_clients):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.rndseed = rndseed,
        self.epochs = epochs,
        

        

        self.clients_max = clients_max,
        self.ip = ip,
        self.dfn_test_y = dfn_test_y,
        self.dfn_test_x = dfn_test_x,
        self.weights = weights,
        self.clf = preprocessing.LabelBinarizer()

        self.DATA_FOLDER = "fl_testbed/version2/data/transformed/"
        self.TRANSFORMED_FOLDER = "fl_testbed/version2/data/transformed/"
        # self.FILE_NAME = "client_independent"
        self.DATA_FILE = "DATASET_"
        self.DATA_FILE2 = "combined_offset_misalignment"
        self.PRECISION = 4

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

        self.df=None
        self.dfn=dfn

    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit) 
        # num_clients = int(num_available_clients * float(self.fq))

        return max(num_clients,
                   self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self,
                               num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        # num_clients = int(num_available_clients * float(self.fq)) 
        

        return max(num_clients,
                   self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(
            self, client_manager: ClientManager) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters,
            client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size,
                                        min_num_clients=min_num_clients)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size,
                                        min_num_clients=min_num_clients)

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [(parameters_to_ndarrays(fit_res.parameters),
                            fit_res.num_examples) for _, fit_res in results]
        parameters_aggregated = ndarrays_to_parameters(
            aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics)
                           for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([(evaluate_res.num_examples,
                                              evaluate_res.loss)
                                             for _, evaluate_res in results])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics)
                            for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(
                eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

    def load_data(self):

        print("INI")

        #TRAINING
        with open(
                self.TRANSFORMED_FOLDER + self.dfn,
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            self.df=pd.read_pickle(file)
            # self.df = pickle.load(file)
        

        #TEST
        with open(
                self.TRANSFORMED_FOLDER + self.dfn_test_x[0].replace('\n', ''),
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            self.X_test = pickle.load(file)

        with open(
                self.TRANSFORMED_FOLDER + self.dfn_test_y[0].replace('\n', ''),
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            self.y_test = pickle.load(file)

        
        
        
        
        
        
        print(self.df.X)
        print(self.df.y)

    def modeling(self):


        print("#ASSIGNAMEMNT")
        df=self.df
        #MANIN ASSIGNAMENT 

        train_inputs=np.array(df.X.apply(lambda x: np.array(x)).tolist())
        # train_inputs=train_inputs
        # self.df.X.apply(lambda row: row[:][:-1])
        train_out=np.array(df.y.apply(lambda x: np.array(x)).tolist())
        # np.array(self.df.y).reshape(-1,1)
        
    
        print(train_inputs)
        print(train_out)

        print(train_inputs.shape)
        print(train_out.shape)
        # input()
        #TESTS PKL
        test_inputs=self.X_test
        test_out=self.y_test
        


        

        # Use the same function above for the validation set WE JUST SPLIT IT IN 0.25 and 0.75 OF THE PREVIOUS SPLIT
        train_inputs, vals_inputs, train_out, vals_out = train_test_split(train_inputs, train_out, 
            test_size=0.25,shuffle=False, random_state= RNDSEED) # 0.25 x 0.8 = 0.2




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
        print(vals_out)
        print(vals_out.shape)
        # polyline = np.array(np.linspace(0,len(vals_out),len(vals_out) ) )
        scaler=MinMaxScaler(feature_range=(0, 1))
        vals_out=scaler.fit_transform(vals_out)
        # vals_out = vals_out.apply(NormalizeData)
        # vals_out = NormalizeData(polyline)
        vals_out=vals_out.reshape(-1,1)
        print(vals_out)
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

        nb_features = train_inputs.shape[2]
        sequence_length  = train_inputs.shape[1]
        nb_out = train_out.shape[1]

        self.model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape = (sequence_length, nb_features), return_sequences = True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(nb_out, activation = 'relu')
        ])

        

        self.model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-7), metrics =['mse','mae'])

        #PASSING VALUES TO THE OTHER FUNCTION!

        
        self.train_inputs=train_inputs
        self.vals_inputs=vals_inputs
        self.vals_out=vals_out
        self.train_out=train_out
        self.test_inputs=test_inputs
        self.test_out=test_out


    def testing(self):

        # print(test_inputs.shape)

        _test_inputs=self.test_inputs[0:]
        _test_out=self.test_out[0:]


        y_pred = self.model.predict(_test_inputs) ## using the untinted dataset!
    
        print('R^2:', metrics.r2_score(_test_out, y_pred))
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(_test_out, y_pred))
        print('Mean Squared Error (MSE):', metrics.mean_squared_error(_test_out, y_pred))
        print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(_test_out, y_pred))
        print('Root Mean Squared Error (RMSE):', np.sqrt(metrics.mean_squared_error(_test_out, y_pred))) # np.sqrt

        print('Explained Variance Score:', metrics.explained_variance_score(_test_out, y_pred))
        print('Max Error:', metrics.max_error(_test_out, y_pred))
        print('Mean Squared Log Error:', metrics.mean_squared_log_error(_test_out, y_pred))
        print('Median Absolute Error:', metrics.median_absolute_error(_test_out, y_pred))
