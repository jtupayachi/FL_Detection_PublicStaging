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

        
        
        
        
        

    def model_definition(self):

        print("#ASSIGNAMEMNT")
        df=self.df

        # X_train=df.X.values
        # y_train=df.y.values
        X_train=np.array(df.X.apply(lambda x: np.array(x)).tolist())#
        # train_inputs=train_inputs
        # self.df.X.apply(lambda row: row[:][:-1])
        y_train=np.array(df.y.apply(lambda x: np.array(x)).tolist())#.

    
        print(X_train)
        print(y_train)

        print(X_train.shape)
        print(y_train.shape)
        # input()
        #TESTS PKL
        X_test=self.X_test
        y_test=self.y_test
        


        

        # Use the same function above for the validation set
        X_train, X_vals, y_train, y_vals = train_test_split(X_train, y_train, 
            test_size=0.25, random_state= RNDSEED,shuffle=True) # 0.25 x 0.8 = 0.2









        from sklearn.preprocessing import StandardScaler,LabelBinarizer
        lbz = LabelBinarizer()

        print("MODELSLSSDLLDSLSD")
        print(X_train.shape)
        print(X_train.dtype)
        print(X_train)

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


        # self.model
        self.model = tf.keras.models.Sequential([
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
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

 
      

        self.X_train=np.array(X_train.astype('float64'))
        self.X_vals=np.array(X_vals.astype('float64'))
        self.y_vals=np.array(y_vals.astype('float64'))
        self.y_train=np.array(y_train.astype('float64'))
        self.X_train=np.array(X_train.astype('float64'))
        self.y_test=np.array(y_test.astype('float64'))


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

        self.history=self.model.fit(self.X_train,self.y_train,validation_data= (self.X_vals,self.y_vals) ,epochs=self.epochs ,verbose=2,callbacks=[tensorboard_callback,lr,es])





    def testing(self):
       
        self.lst_accu_stratified_a=max(self.history.history['val_accuracy'])
        #https://datascience.stackexchange.com/questions/116692/accuracy-vs-categorical-accuracy
        # metrics: List of metrics to be evaluated by the model during training and testing. Each of this can be a string (name of a built-in function), function or a tf.keras.metrics.Metric instance. See tf.keras.metrics. Typically you will use metrics=['accuracy']. A function is any callable with the signature result = fn(y_true, y_pred). To specify different metrics for different outputs of a multi-output model, you could also pass a dictionary, such as metrics={'output_a':'accuracy', 'output_b':['accuracy', 'mse']}. You can also pass a list to specify a metric or a list of metrics for each output, such as metrics=[['accuracy'], ['accuracy', 'mse']] or metrics=['accuracy', ['accuracy', 'mse']]. When you pass the strings 'accuracy' or 'acc', we convert this to one of tf.keras.metrics.BinaryAccuracy, tf.keras.metrics.CategoricalAccuracy, tf.keras.metrics.SparseCategoricalAccuracy based on the shapes of the targets and of the model output. We do a similar conversion for the strings 'crossentropy' and 'ce' as well. The metrics passed here are evaluated without sample weighting; if you would like sample weighting to apply, you can specify your metrics via the weighted_metrics argument instead.

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
