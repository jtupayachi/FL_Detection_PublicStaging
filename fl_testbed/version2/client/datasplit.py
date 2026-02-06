#!/usr/bin/env python3 #
"""
    This script splits the data in an specified number of clients es. 0,1,2,3,4...
    The number of clients must equal to the max number of clients. Take into account that the client number is specified from 0,
    and the max number of clients start from 1.

    Two main parameters are:
    -l  200 3 30 2 200 10 40 150 90 10 60 15 89 70 10 100 10 35 10 90 :: It is defined as the manual weights for the NIID data split.
    -fq 0.1 0.6 0.3 1 :: It is defined as the fraction of the residual dataframe, the last dataframe must be 1 to use all the data in
    the provided dataframe.     

    
    //NEW WAY TO MAKE IT OWRK
    RUN IT THREE TIMES WITH SPECIFIC _Mx csvs files 


"""
import sys
import pickle
import numpy as np
from collections.abc import Iterable
import pandas as pd
import os
import itertools
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import argparse
import numpy as np


class DataSplit:

    def __init__(
        self,
        data_X_train,
        data_X_vals,
        data_y_train,
        data_y_vals,
        # rndseed,
        clients_max, #NUMBER FROM 0 to n clients
        manual_list, #LIST TO PROVIDE
        fraq, #FRACTION OF EACH 
        motor, #IDENTIFIER M1,M2,M3
        type,
    ):

        self.data_X_train=data_X_train #PATH
        self.data_X_vals=data_X_vals #PATH
        self.data_y_train=data_y_train #PATH
        self.data_y_vals=data_y_vals #PATH
        # self.rndseed = rndseed
        self.clients_max = clients_max
        self.manual_list = (manual_list, )
        self.fraq = fraq
        self.motor = motor[0]
        self.type = type[0]

        
        






        self.concatenated_identifier = (
                                        str(self.motor) + "_" +

                                        str(self.clients_max[0]) + "_")
        


        self.DATA_FOLDER = "fl_testbed/version2/data/transformed/"
        self.TRANSFORMED_FOLDER = "fl_testbed/version2/data/transformed/"
        self.FILE_NAME = "datasplit"
        self.PRECISION = 4


        self.X=None
        self.y=None


    def load_data_1(self):

        with open(self.DATA_FOLDER+self.data_X_train[0], 'rb') as f: 
            X_train = pd.read_pickle(f)

        #X_VALS
        with open(self.DATA_FOLDER+self.data_X_vals[0], 'rb') as f: 
            X_vals = pd.read_pickle(f)

        #y_TRAIN
        with open(self.DATA_FOLDER+self.data_y_train[0], 'rb') as f: 
            y_train = pd.read_pickle(f)

        #y_VALS
        with open(self.DATA_FOLDER+self.data_y_vals[0], 'rb') as f: 
            y_vals = pd.read_pickle(f)




        y_train=y_train.values.reshape(-1,1)
        y_vals=y_vals.values.reshape(-1,1)

        print(X_train.shape)
        print(X_vals.shape)
        print(y_train.shape)
        print(y_vals.shape)

        self.X=np.vstack((X_train,X_vals))
        self.y=np.vstack((y_train,y_vals))
        # print(self.X)
        # print(self.y)
        print(self.X.shape)
        print(self.y.shape)


        df_X = pd.DataFrame.from_records(zip(self.X),columns=['X'])
        df_y = pd.DataFrame.from_records(zip(self.y),columns=['y'])


        self.df=df_X.merge(df_y,left_index=True,right_index=True)
        

    def load_data_2(self):

        with open(self.DATA_FOLDER+self.data_X_train[0], 'rb') as f: 
            X_train = pickle.load(f)

        #X_VALS
        with open(self.DATA_FOLDER+self.data_X_vals[0], 'rb') as f: 
            X_vals = pickle.load(f)

        #y_TRAIN
        with open(self.DATA_FOLDER+self.data_y_train[0], 'rb') as f: 
            y_train = pickle.load(f)

        #y_VALS
        with open(self.DATA_FOLDER+self.data_y_vals[0], 'rb') as f: 
            y_vals = pickle.load(f)



        print(X_train.shape)
        print(X_vals.shape)
        print(y_train.shape)
        print(y_vals.shape)

        self.X=np.vstack((X_train,X_vals))
        self.y=np.vstack((y_train,y_vals))
        # print(self.X)
        # print(self.y)
        print(self.X.shape)
        print(self.y.shape)


        df_X = pd.DataFrame.from_records(zip(self.X),columns=['X'])
        df_z = pd.DataFrame.from_records(zip(self.y),columns=['z'])
        
        ##NEED TO WORK


        self.df=df_X.merge(df_z,left_index=True,right_index=True)
        self.df['y'] = int(self.df.X[0][0][-1])
        print(self.df)
        # print(self.df.y.values)
        


   
    def splitdata_1(self):

        # self.create_histogram(self.df, "TOTAL_DATASET",
        #                       self.TRANSFORMED_FOLDER + "M"+str(self.motor)+"_TOTAL_DATASET")

        print(self.df.y)

        col = "y"
        
        unique_values = set(self.df[col].apply(tuple))
        
        order = list(unique_values)
        
        
        print("ORDER:", len(order))
        
        # print(manual_list[0])
        # print(manual_list)
        manual_list_clustered = [manual_list[0][i:i + int(len(order))] for i in range(0, len(manual_list[0]), int(len(order)))]

        # [list(manual_list) for k, it in groupby(x_sorted, self.clients_max)]  #[manual_list[x:x+self.clients_max] for x in range(0,len(manual_list),self.clients_max)] #[self.manual_list[i:i+int(self.clients_number)] for i in range(0,len(self.manual_list),int(self.clients_number))]
        print(manual_list_clustered)

        df = self.df.copy()
        counter = 0

        df.reset_index(inplace=True)

        for choices, f in zip(manual_list_clustered, fraq[0]):

            print(df.shape)
            print("counter", counter)

            conditions = []
            for o in order:
                print(o)
                conditions.append(df[col].apply(tuple) == o)

            
            # choices = i
            print(conditions)
            print(len(conditions))
            print("######")
            print(choices)
            print(len(choices))
            df["weights"] = np.select(conditions, choices, default=np.nan)

            df.set_index("index")
            
            sample_df = df.sample(frac=float(f),
                                  weights="weights",
                                  random_state=RNDSEED)
            
            sample_df.set_index("index")









            sample_index = sample_df.index

            sample_df_tosave = sample_df[sample_df.columns.drop(list(sample_df.filter(regex="weights")))]
                


            sample_df_tosave.drop(columns='index',axis=0,inplace=True)

            # X=sample_df_tosave['X']
            y=sample_df_tosave['y']


            #COUTNER!!!
            y_unested=[val for sublist in y for val in sublist]
            unique, counts = np.unique(y_unested, return_counts=True)
            print(unique)
            print(counts)
            print("COUNTER: ",dict(zip(unique, counts)))
            #OK

                
            #HERE WE SAVE SMALL DATASETS!!!
            print("concatenated_identifier")
            print(self.concatenated_identifier)
            with open(
                    self.TRANSFORMED_FOLDER + "M"+self.concatenated_identifier +str(counter)+"_" +
                    self.FILE_NAME[0] + "df_MLP.pkl",
                    "wb",
            ) as file:
                pickle.dump(sample_df_tosave, file)

            # sample_df_tosave.to_csv(self.TRANSFORMED_FOLDER +str(self.type)+"_M"+str(self.motor)+"_"+ "DATASET_" + str(counter) +".csv",index=False,)    
                
            

            # self.create_histogram(sample_df,"CLIENT1",'status')
            # AFTER CREATING HISTOMGRAM
            sample_df = sample_df.drop("weights", axis=1)

            # again assigned to df and split again.
            df = df.loc[~df.index.isin(sample_index)]
            counter = counter + 1


    def splitdata_2(self):

        print(self.df.y)

        col = "y"
        
        unique_values = set(self.df[col])
        
        order = list(unique_values)
        
        
        print("ORDER:", len(order))


        manual_list_clustered = [manual_list[0][i:i + int(len(order))] for i in range(0, len(manual_list[0]), int(len(order)))]

        # [list(manual_list) for k, it in groupby(x_sorted, self.clients_max)]  #[manual_list[x:x+self.clients_max] for x in range(0,len(manual_list),self.clients_max)] #[self.manual_list[i:i+int(self.clients_number)] for i in range(0,len(self.manual_list),int(self.clients_number))]
        print(manual_list_clustered)

        df = self.df.copy()
        counter = 0

        df.reset_index(inplace=True)

        for choices, f in zip(manual_list_clustered, fraq[0]):

            # print(df.shape)
            print("counter", counter)

            conditions = []
            for o in order:
                # print(o)
                conditions.append(df[col] == o)

            
            df["weights"] = np.select(conditions, choices, default=np.nan)

            df.set_index("index")
            
            sample_df = df.sample(frac=float(f),
                                  weights="weights",
                                  random_state=RNDSEED)
            
            sample_df.set_index("index")

            sample_index = sample_df.index

            sample_df_tosave = sample_df[sample_df.columns.drop(list(sample_df.filter(regex="weights")))]
                
            sample_df_tosave.rename(columns={'y':'to_drop'},inplace=True)
            sample_df_tosave.rename(columns={'z':'y'},inplace=True)

            sample_df_tosave.drop(columns=['index','to_drop'],axis=0,inplace=True)

            # X=sample_df_tosave['X']
            y=sample_df_tosave['y']

            #COUTNER!!!
            y_unested=[val for sublist in y for val in sublist]
            unique, counts = np.unique(y_unested, return_counts=True)

            #NEW EDITED LINE
            print(unique)
            print(counts)
            print("COUNTER: ",dict(zip(unique, counts)))
            #NEW EDITED LINE
                
            #HERE WE SAVE SMALL DATASETS!!!
            # print("concatenated_identifier")
            print(self.concatenated_identifier)
            with open(
                    self.TRANSFORMED_FOLDER + "M"+self.concatenated_identifier +str(counter)+"_" +
                    self.FILE_NAME[0] + "df_LSTM.pkl",
                    "wb",
            ) as file:
                pickle.dump(sample_df_tosave, file)

            # sample_df_tosave.to_csv(self.TRANSFORMED_FOLDER +str(self.type)+"_M"+str(self.motor)+"_"+ "DATASET_" + str(counter) +".csv",index=False,)    
            # self.create_histogram(sample_df,"CLIENT1",'status')
            # AFTER CREATING HISTOMGRAM
            sample_df = sample_df.drop("weights", axis=1)

            # again assigned to df and split again.
            df = df.loc[~df.index.isin(sample_index)]
            counter = counter + 1




if __name__ == "__main__":



    # Arguments
    parser = argparse.ArgumentParser()





    parser.add_argument(
        "-data_X_train",
        "--data_X_train",
        help="data_X_train",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_X_vals",
        "--data_X_vals",
        help="data_X_vals",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_y_train",
        "--data_y_train",
        help="data_y_train",
        type=str,
        required=True,
    ),
    parser.add_argument(
        "-data_y_vals",
        "--data_y_vals",
        help="data_y_vals",
        type=str,
        required=True,
    )
   
    parser.add_argument(
        "-cm",
        "--clients_max",
        help="Maximun number of clients",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-l",
        "--manual_list",
        help="manual List of label distribution",
        nargs="+",
        required=True,
        default=None,
    )
    parser.add_argument("-fq",
                        "--fraq",
                        help="fraction list",
                        nargs="+",
                        required=True,
                        default=None)

    parser.add_argument("-motor",
                    "--motor",
                    help="motor number",
                    type=int,
                    required=True,
                    default=None)
    
    parser.add_argument("-type",
                "--type",
                help="type",
                type=str,
                required=True,
                default=None)










    args = parser.parse_args()




    data_X_train=str(args.data_X_train),
    data_X_vals=str(args.data_X_vals),
    data_y_train=str(args.data_y_train),
    data_y_vals=str(args.data_y_vals),
    
    clients_max=int(args.clients_max), #NUMBER FROM 0 to n clients
    manual_list=args.manual_list, #LIST TO PROVIDE
    fraq=args.fraq, #FRACTION OF EACH 
    motor=args.motor, #IDENTIFIER M1,M2,M3
    type=args.type,
    
   

  

    # Configuration
    root_path = os.path.dirname(os.path.abspath("__file__"))
    os.chdir(root_path)

    RNDSEED = np.random.seed(39)
    np.random.seed(RNDSEED)

    os.environ["PYTHONHASHSEED"] = str(RNDSEED)
    
    datasplit = DataSplit(

        data_X_train,
        data_X_vals,
        data_y_train,
        data_y_vals,

        clients_max, #NUMBER FROM 0 to n clients
        manual_list, #LIST TO PROVIDE
        fraq, #FRACTION OF EACH 
        motor, #IDENTIFIER M1,M2,M3
        type,
 
    )

    
    if type[0] == 'MLP':
        print("ENTERED!")
        datasplit.load_data_1()
        print(type[0])
        datasplit.splitdata_1()
    if type[0] == 'LSTM':
        print("ENTERED2!")
        datasplit.load_data_2()
        print(type[0])
        datasplit.splitdata_2()
    
    exit()
