"""Flower client app.

This script packs a set of functions for performing the federated learning steps. Functions like get_parameters,
fit_parameters are given by flwr. This script has been modified to include in the initialization class
different models and efficiently change of their parameters.

"""
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


#MLP








from sklearn.model_selection import train_test_split

# from autosklearn.regression import AutoSklearnRegressor

from tensorflow.keras.regularizers import L2,L1,L1L2











#LSTM












from sklearn.model_selection import train_test_split

# from autosklearn.regression import AutoSklearnRegressor




import pandas as pd
import os
import numpy as np
RNDSEED = np.random.seed(39)
PRECISION = 4 # 3 of digits to keep after the decimal point





from sklearn.preprocessing import StandardScaler, MinMaxScaler
RUNNING_ON_COLAB = False # we assume running on CoLab! Change to False if running locally.










# Libraries





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




class NumPyClient(ABC):
    """Abstract base class for Flower clients using NumPy."""

    def __init__(
        self,
        data_file_name,
        dfn_test_x,
        dfn_test_y,
        rndseed,
        epochs,
        
        
        clients_max,
        clients_number,
        ip,
    ) -> None:

        super().__init__()

        self.data_file_name = data_file_name
        self.rndseed = rndseed
        self.epochs = epochs


        self.clients_max = clients_max
        self.clients_number = clients_number
        self.ip = ip
        self.dfn_test_x=dfn_test_x
        self.dfn_test_y=dfn_test_y

        self.concatenated_identifier = (str(self.epochs) + "_" +

                                        str(self.clients_max) + "_" +
                                        str(self.clients_number) + "_" +
                                        str(self.data_file_name) + "__")
        self.DATA_FOLDER = "fl_testbed/version2/data/transformed/"
        self.TRANSFORMED_FOLDER = "fl_testbed/version2/data/transformed/"
        self.FILE_NAME = "client_federated"
        self.DATA_FILE = "DATASET_"
        self.DATA_FILE2 = "combined_offset_misalignment"
        self.PRECISION = 4
        self.y_train = None
        self.X_train = None
        self.y_test = None
        self.X_test = None
        self.df = None
        # self.model = None
        self.X = None
        self.y = None
        self.clf = preprocessing.LabelBinarizer()
        self.skf = None
        self.es = None
        self.k_fold_separation = None
        self.lst_accu_stratified_a = None
        self.lst_accu_stratified_l = None
        self.history=None
                
        self.train_inputs=None
        self.vals_inputs=None
        self.vals_out=None
        self.train_out=None
        self.test_inputs=None
        self.test_out=None

    def get_properties(self, config: Config) -> Dict[str, Scalar]:
        """Returns a client's set of properties.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which properties
            are needed along with some Scalar attributes.

        Returns
        -------
        properties : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary property values back to the server.
        """

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current local model parameters.

        Parameters
        ----------
        config : Config
            Configuration parameters requested by the server.
            This can be used to tell the client which parameters
            are needed along with some Scalar attributes.

        Returns
        -------
        parameters : NDArrays
            The local model parameters as a list of NumPy ndarrays.
        """

    def fit(self, parameters: NDArrays,
            config: Dict[str,
                         Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the
            server to influence training on the client. It can be used to
            communicate arbitrary values from the server to the client, for
            example, to set the number of (local) training epochs.

        Returns
        -------
        parameters : NDArrays
            The locally updated model parameters.
        num_examples : int
            The number of examples used for training.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of type
            bool, bytes, float, int, or str. It can be used to communicate
            arbitrary values back to the server.
        """

    def evaluate(
            self, parameters: NDArrays,
            config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the provided parameters using the locally held dataset.

        Parameters
        ----------
        parameters : NDArrays
            The current (global) model parameters.
        config : Dict[str, Scalar]
            Configuration parameters which allow the server to influence
            evaluation on the client. It can be used to communicate
            arbitrary values from the server to the client, for example,
            to influence the number of examples used for evaluation.

        Returns
        -------
        loss : float
            The evaluation loss of the model on the local dataset.
        num_examples : int
            The number of examples used for evaluation.
        metrics : Dict[str, Scalar]
            A dictionary mapping arbitrary string keys to values of
            type bool, bytes, float, int, or str. It can be used to
            communicate arbitrary values back to the server.

        Warning
        -------
        The previous return type format (int, float, float) and the
        extended format (int, float, float, Dict[str, Scalar]) have been
        deprecated and removed since Flower 0.19.
        """

    def load_data(self):
        
        print("INI")

        #TRAINING
        with open(
                self.TRANSFORMED_FOLDER + self.data_file_name,
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            # self.df = pickle.load(file)
            self.df=pd.read_pickle(file)
        

        #TEST
        with open(
                self.TRANSFORMED_FOLDER + self.dfn_test_x.replace('\n', ''),
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            self.X_test = pickle.load(file)

        with open(
                self.TRANSFORMED_FOLDER + self.dfn_test_y.replace('\n', ''),
                # self.concatenated_identifier +
                # self.FILE_NAME + "y_train.pkl",
                "rb",
        ) as file:
            self.y_test = pickle.load(file)

        
        
        
        
        
        
        print(self.df.X)
        print(self.df.y)

 
    def model_definition(self):

        print("#ASSIGNAMEMNT")
        df=self.df

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


    def modeling(self):


        lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

        # model.compile(loss=tf.keras.losses.Huber(), optimizer = tf.keras.optimizers.Adam(learning_rate = 10**-7), metrics =['mse','mae'])


        #FAST AI SEE IF TRIANING IMPROVES !

        es = EarlyStopping(monitor="val_loss",
                mode="auto",
                verbose=2,
                patience=1,
                min_delta=0.001,
                restore_best_weights=True)
        # 1420492
        # history = model.fit(train_inputs, train_out, epochs = 20, callbacks = [lr])
        lr = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**-7 * 10**(epoch/3))

        from  datetime import datetime

        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        print(log_dir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


        self.history=self.model.fit(self.train_inputs,self.train_out,epochs=self.epochs,validation_data= (self.vals_inputs,self.vals_out) ,verbose=2,callbacks=[tensorboard_callback,lr,es],)





    def testing(self):

        #FOR FLWR:
        self.lst_accu_stratified_a=min(self.history.history['val_mse'])
        self.lst_accu_stratified_l=min(self.history.history['val_loss'])

      

def has_get_properties(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_properties."""
    return type(client).get_properties != NumPyClient.get_properties


def has_get_parameters(client: NumPyClient) -> bool:
    """Check if NumPyClient implements get_parameters."""
    return type(client).get_parameters != NumPyClient.get_parameters


def has_fit(client: NumPyClient) -> bool:
    """Check if NumPyClient implements fit."""
    return type(client).fit != NumPyClient.fit


def has_evaluate(client: NumPyClient) -> bool:
    """Check if NumPyClient implements evaluate."""
    return type(client).evaluate != NumPyClient.evaluate
