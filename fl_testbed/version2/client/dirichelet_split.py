#SIMILAR IMPLEMENTATION AS: arXiv:1909.06335
#FOR client_max use 5 for 5 cleints it iwll create 0,1,2,3,4 , use 4 for 4 clients it wil create 0,1,2,3  


"""
    NEW IMPLEMENTATION!

    MLP M3: python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_train.pkl -data_X_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedX_vals.pkl -data_y_train 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_train.pkl -data_y_vals 100_1_15_15_combined_offset_misalignment_M3.csv__client_centralizedy_vals.pkl -cm 5 -alpha 0.02 -beta 0.2 -motor 3 -type MLP 
   
   LSTM M3: python3 fl_testbed/version2/client/dirichelet_split.py -data_X_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_inputs.pkl -data_X_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_inputs.pkl -data_y_train 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedtrain_out.pkl -data_y_vals 100_2_15_15_combined_offset_misalignment_M3.csv__client_centralizedvals_out.pkl -cm 5 -alpha 0.02 -beta 0.2 -motor 3 -type LSTM

"""

import random
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
import scipy
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from collections import Counter


# Set a random seed for NumPy
np.random.seed(42)




class DataSplit:

    def __init__(
        self,
        data_X_train,
        data_X_vals,
        data_y_train,
        data_y_vals,
        clients_max, #NUMBER FROM 0 to n clients, IDENTIFIER: 0,1,2,3,4 ....
        alpha, #DIRICHELET ALPHA
        beta, #PARAMETER SPLIT SIZE PER CLIENT/CLASS
        motor, #IDENTIFIER M1,M2,M3
        type,
    ):

        self.data_X_train=data_X_train #PATH
        self.data_X_vals=data_X_vals #PATH
        self.data_y_train=data_y_train #PATH
        self.data_y_vals=data_y_vals #PATH
        self.clients_max = clients_max[0]
        self.alpha=alpha[0]
        self.beta=beta[0]
        self.motor = motor[0]
        self.type = type[0]

        
        self.concatenated_identifier = (
                                        str(self.motor) + "_" +

                                        str(self.clients_max) + "_")
        

        self.DATA_FOLDER = "fl_testbed/version2/data/transformed/"
        self.TRANSFORMED_FOLDER = "fl_testbed/version2/data/transformed/"
        self.FILE_NAME = "dirichelet_split"
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




        # y_train=y_train.values.reshape(-1,1)
        # y_vals=y_vals.values.reshape(-1,1)

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

        print("PRINTING DATAFRME")
        print(self.df.tail())
        # print(df_X..value_counts())
        print("###############################")



        # self.df['y'] = int(self.df.X[0][0][-1]) THIS ONLY TAKE THE FIRST ELEMENT AND REPRODUCES IT 
        self.df['y'] = int(self.df.X[0][0][-1])

        # Define a function to access the last element of a NumPy array
        def get_last_element(arr):
            return int(arr[-1, -1])

        # Apply the function to the DataFrame
        self.df['y'] = self.df['X'].apply(get_last_element)



        print("PRINTING DATAFRME")
        print(self.df.tail())
        # print(df_X..value_counts())
        print("###############################")


        
        # print(self.df.y.values)
        
    def splitdata_1(self):

        print(self.df.head(5))

        #DATA_FILE 
        # print(self.df.y.value_counts()[0]) CAN BE ACCESSED
        print(self.df.y.value_counts())

        
        #ALPHA
        print(self.alpha)
        #BETA
        print(self.beta)
        #CM
        print(self.clients_max)

        print("END OF READING!")

        # Example usage:
        N = self.clients_max # Number of classes
        alpha = self.alpha  # Concentration parameter (a scalar)
        num_clients = self.clients_max
        num_samples_per_client = 100000000 #UYSE A HIGH NUMBER FOR USE ALL

        p = [self.beta] * N  # Prior class distribution (1-dimensional) STARTS WITH SAME NUMBER OF DATAPOITNS PER PARTITION
        
        
        total = sum(p)
        # Standardize the numbers to sum to 1
        p = [x / total for x in p]

        print("STANDARIZED : ",p)

        print(os.getcwd)

        def split_dataframe_to_non_identical_clients(df, N, alpha, num_clients, num_samples_per_client, p, output_dir):



            def sample_group(group, frac):
                return group.sample(frac=frac)

            def unwrap_array(arr):
                return arr[0] 


            if alpha <= 0:
                raise ValueError("Alpha must be a positive number.")
            if len(p) != N or not all(0 <= qi <= 1 for qi in p):
                raise ValueError("The prior class distribution 'p' must be a 1-dimensional array with values between 0 and 1.")
            
            clients = []
            # df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
            
            # print(p)
            print("flattened_MLP")
            print(df['y'])
            flattened_y = np.concatenate(df['y'])

            # Get unique classes
            unique_classes = np.unique(flattened_y)
            # Convert the unique classes to a list if needed
            unique_classes_list = unique_classes.tolist()
            

            value_counts = df['y'].value_counts()
            min_count_class = value_counts.idxmin()[0]
            min_count_quantity = value_counts.min()

            
            counter=0
            aggregated_probabilities=[]
            #FOR EACH OF MY 0.2 SPLITS OF DATA!
            for i in p:

                print("This is split: "+str(counter))
                
                #DATAFRAME CREATION!

                #THIS SECTION IS FOR CREATING THE EUQLA DATA PARTITIIONS
                # Group the DataFrame by a custom key, e.g., a constant, since we want to sample without grouping
                grouped = self.df.groupby(lambda x: 0)

                # Use apply to sample rows within each group
                sampled_df = grouped.apply(sample_group, frac=i)

    
                #DISTRIBUTION GENENRATION!
                num_samples = min(num_samples_per_client, len(df))
                data = df.iloc[:num_samples, :].copy()
                df = df.iloc[num_samples:, :]

                q = dirichlet.rvs([alpha] * N, random_state=42)  # Ensure alpha is a scalar
                q = np.maximum(q, 0)
                q /= q.sum()




                #ORDENING!!

                # Find the index of the greatest element in the array
                greatest_index = np.argmax(q)

                # Check if the counter matches the position of the greatest element
                if counter != greatest_index:
                    # Rearrange the array so that the counter matches the greatest element's position
                    rearranged_items = np.roll(q, counter - greatest_index)
                    q = rearranged_items
                #ROUNDING P

                
                aggregated_probabilities.append(np.squeeze(q))

                #MY MAPPER! FOR EACH DATASPLIT !
                # class_probabilities=zip(unique_classes_list,np.squeeze(q))
                result_dict = [{'map': item[0], 'weights': item[1]} for item in zip(unique_classes_list,np.squeeze(q))]

                # If you want to convert the result_dict into a DataFrame:
                class_probabilities= pd.DataFrame(result_dict)
                class_probabilities['map'] = class_probabilities['map'].apply(lambda x: np.array([x]))

                # Map the DataFrames using the custom function
                sampled_df['mapper'] = sampled_df['y'].apply(unwrap_array)
                class_probabilities['mapper'] = class_probabilities['map'].apply(unwrap_array)

                merged_df = pd.merge(sampled_df, class_probabilities, on='mapper',how='inner')
                # print("FIRST")
                # print(merged_df.mapper.value_counts().sort_index())

                


                #CREATE ANOTHER DATAFRAME JUST FOR THE MINIMUN OF 30 DATAPOITNS TO ENSURE ALL CLASSES ARE REPRESENTED
                min_data_points_per_class = 400



 
                # Create a DataFrame to store the selected data points for each class
                selected_data = pd.DataFrame()


                # Iterate through unique classes
                for class_label in merged_df['mapper'].unique():
                    class_data = merged_df[merged_df['mapper'] == class_label]
                    
                    # If there are fewer than 10 data points in the class, select them all
                    if len(class_data) < min_data_points_per_class:
                        selected_data = pd.concat([selected_data, class_data])
                    else:
                        # Randomly select 10 data points from the class
                        selected_data = pd.concat([selected_data, class_data.sample(n=min_data_points_per_class, replace=False)])


                # Randomly sample the remaining data points if necessary
                merged_df = merged_df[~merged_df.index.isin(selected_data.index)]

                # Calculate the sum of the specified column
                column_sum = merged_df['weights'].sum()

                # Normalize the values in the column so that they sum up to 1
                merged_df['weights'] = merged_df['weights'] / column_sum

                # Extract the weights as a NumPy array
                weights = merged_df['weights'].values

                # Perform a random choice based on the weights
                # Perform random sampling without replacement
                sampled_index = random.choices(merged_df.index, weights=weights, k=min_count_quantity)

                # Get the selected row(s) from the DataFrame
                merged_df_sampled = merged_df.loc[sampled_index]
                
                # merged_df.sample(n=min_count_quantity, weights=weights, replace=False)

                # Combine the selected data from each class with the sampled remaining data
                merged_df_sampled = pd.concat([selected_data, merged_df_sampled])

                print(merged_df_sampled.mapper.value_counts().sort_index())

                #SAVING DATAFRAME!
                #NEW EDITED LINE

                merged_df_sampled=merged_df_sampled[['X','y']]
                    
                #HERE WE SAVE SMALL DATASETS!!!
                print(self.concatenated_identifier)
                with open(
                        self.TRANSFORMED_FOLDER + "M"+self.concatenated_identifier +str(counter)+"_" +
                        self.FILE_NAME[0] + "df_MLP.pkl",
                        "wb",
                ) as file:
                    pickle.dump(merged_df_sampled, file)

                counter=counter+1
            
            to_plot=pd.DataFrame(aggregated_probabilities)#.add_prefix()
            print(to_plot)
            plot_class_distribution_and_save(to_plot,N)
            


            return clients

        def plot_class_distribution_and_save(to_plot,N):
            data=to_plot
            bar_width = 0.2  # Width of each bar
            num_rows = len(data)
            x = np.arange(len(data[0]))  # X-axis values


            fig, ax = plt.subplots()

            # Initialize a variable to keep track of the bottom position for each bar
            bottom = np.zeros(N)

            for i in range(num_rows):
                ax.bar(x, data[i], label=f'Error Type {i + 1}', bottom=bottom)
                bottom += data[i]  # Update the bottom position for the next row

            ax.set_xlabel('Client')
            ax.set_ylabel('Class Probability')
            
            ax.set_ylim(0, 1)

            ax.set_title("\u03B1 = "+ str(self.alpha))

            # Set the ticks and labels
            ax.set_xticks(x)
            # ax.set_xticklabels(['Column 1', 'Column 2', 'Column 3'])

            # Add a legend
            ax.legend()
            

            plt.savefig(self.TRANSFORMED_FOLDER + "M"+self.concatenated_identifier +"_" +
                        self.FILE_NAME[0] + "df_MLP_"+str(self.alpha)+"_PLOT.pdf")

        print("SPLITS!!!")


        # Output directory for saving plots
        output_directory = "client_plots"
        os.makedirs(output_directory, exist_ok=True)

        non_identical_clients = split_dataframe_to_non_identical_clients(self.df, N, alpha, num_clients, num_samples_per_client, p, output_directory)
        print(non_identical_clients)














    def splitdata_2(self):

        def wrap_with_numpy_array(value):
            return np.array([value])

        
        print("First!!")
        print(self.df.y.value_counts())

        #ONLY FOR LSTM
        self.df['y'] = self.df['y'].apply(wrap_with_numpy_array)


        print(self.df.head(5))


        #DATA_FILE 
        # print(self.df.y.value_counts()[0]) CAN BE ACCESSED
        print(self.df.y.value_counts())

        
        #ALPHA
        print(self.alpha)
        #BETA
        print(self.beta)
        #CM
        print(self.clients_max)

        print("END OF READING!")

        # Example usage:
        N = self.clients_max # Number of classes
        alpha = self.alpha  # Concentration parameter (a scalar)
        num_clients = self.clients_max
        num_samples_per_client = 100000000 #UYSE A HIGH NUMBER FOR USE ALL

        p = [self.beta] * N  # Prior class distribution (1-dimensional) STARTS WITH SAME NUMBER OF DATAPOITNS PER PARTITION
        
        
        total = sum(p)
        # Standardize the numbers to sum to 1
        p = [x / total for x in p]

        print("STANDARIZED : ",p)


        def split_dataframe_to_non_identical_clients(df, N, alpha, num_clients, num_samples_per_client, p, output_dir):

            



            def sample_group(group, frac):
                return group.sample(frac=frac)

            def unwrap_array(arr):
                return arr[0] 




            if alpha <= 0:
                raise ValueError("Alpha must be a positive number.")
            if len(p) != N or not all(0 <= qi <= 1 for qi in p):
                raise ValueError("The prior class distribution 'p' must be a 1-dimensional array with values between 0 and 1.")
            
            clients = []
            # df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the DataFrame
            


            # print(p)
            print(df['y'])
            print("flattened_LSTM")
            
            

            flattened_y = np.concatenate(df['y'].values)
            # Get unique classes
            unique_classes = np.unique(flattened_y)
            # Convert the unique classes to a list if needed
            unique_classes_list = unique_classes.tolist()
            

            value_counts = df['y'].value_counts()
            min_count_class = value_counts.idxmin()[0]
            min_count_quantity = value_counts.min()

            
            counter=0
            aggregated_probabilities=[]
            #FOR EACH OF MY 0.2 SPLITS OF DATA!
            for i in p:

                print("This is split: "+str(counter))
                
                #DATAFRAME CREATION!

                #THIS SECTION IS FOR CREATING THE EUQLA DATA PARTITIIONS
                # Group the DataFrame by a custom key, e.g., a constant, since we want to sample without grouping
                grouped = self.df.groupby(lambda x: 0)

                # Use apply to sample rows within each group
                sampled_df = grouped.apply(sample_group, frac=i)

    
                #DISTRIBUTION GENENRATION!
                num_samples = min(num_samples_per_client, len(df))
                data = df.iloc[:num_samples, :].copy()
                df = df.iloc[num_samples:, :]

                q = dirichlet.rvs([alpha] * N, random_state=42)  # Ensure alpha is a scalar
                q = np.maximum(q, 0)
                q /= q.sum()




                #ORDENING!!

                # Find the index of the greatest element in the array
                greatest_index = np.argmax(q)

                # Check if the counter matches the position of the greatest element
                if counter != greatest_index:
                    # Rearrange the array so that the counter matches the greatest element's position
                    rearranged_items = np.roll(q, counter - greatest_index)
                    q = rearranged_items
                #ROUNDING P

                
                aggregated_probabilities.append(np.squeeze(q))

                #MY MAPPER! FOR EACH DATASPLIT !
                # class_probabilities=zip(unique_classes_list,np.squeeze(q))
                result_dict = [{'map': item[0], 'weights': item[1]} for item in zip(unique_classes_list,np.squeeze(q))]

                # If you want to convert the result_dict into a DataFrame:
                class_probabilities= pd.DataFrame(result_dict)
                class_probabilities['map'] = class_probabilities['map'].apply(lambda x: np.array([x]))

                # Map the DataFrames using the custom function
                sampled_df['mapper'] = sampled_df['y'].apply(unwrap_array)
                class_probabilities['mapper'] = class_probabilities['map'].apply(unwrap_array)

                merged_df = pd.merge(sampled_df, class_probabilities, on='mapper',how='inner')
                # print("FIRST")
                # print(merged_df.mapper.value_counts().sort_index())

                


                #CREATE ANOTHER DATAFRAME JUST FOR THE MINIMUN OF 30 DATAPOITNS TO ENSURE ALL CLASSES ARE REPRESENTED
                min_data_points_per_class = 400



 
                # Create a DataFrame to store the selected data points for each class
                selected_data = pd.DataFrame()


                # Iterate through unique classes
                for class_label in merged_df['mapper'].unique():
                    class_data = merged_df[merged_df['mapper'] == class_label]
                    
                    # If there are fewer than 10 data points in the class, select them all
                    if len(class_data) < min_data_points_per_class:
                        selected_data = pd.concat([selected_data, class_data])
                    else:
                        # Randomly select 10 data points from the class
                        selected_data = pd.concat([selected_data, class_data.sample(n=min_data_points_per_class, replace=False)])


                # Randomly sample the remaining data points if necessary
                merged_df = merged_df[~merged_df.index.isin(selected_data.index)]

                # Calculate the sum of the specified column
                column_sum = merged_df['weights'].sum()

                # Normalize the values in the column so that they sum up to 1
                merged_df['weights'] = merged_df['weights'] / column_sum

                # Extract the weights as a NumPy array
                weights = merged_df['weights'].values

                # Perform a random choice based on the weights
                # Perform random sampling without replacement
                sampled_index = random.choices(merged_df.index, weights=weights, k=min_count_quantity)

                # Get the selected row(s) from the DataFrame
                merged_df_sampled = merged_df.loc[sampled_index]
                
                # merged_df.sample(n=min_count_quantity, weights=weights, replace=False)

                # Combine the selected data from each class with the sampled remaining data
                merged_df_sampled = pd.concat([selected_data, merged_df_sampled])

                print(merged_df_sampled.mapper.value_counts().sort_index())

                #SAVING DATAFRAME!
                merged_df_sampled=merged_df_sampled[['X','z']]

                merged_df_sampled.rename(columns={'z': 'y'},inplace=True)
                
                print(merged_df_sampled)
                #NEW EDITED LINE
                    
                #HERE WE SAVE SMALL DATASETS!!!
                print(self.concatenated_identifier)
                with open(
                        self.TRANSFORMED_FOLDER + "M"+self.concatenated_identifier +str(counter)+"_" +
                        self.FILE_NAME[0] + "df_LSTM.pkl",
                        "wb",
                ) as file:
                    pickle.dump(merged_df_sampled, file)

                counter=counter+1
            
            to_plot=pd.DataFrame(aggregated_probabilities)#.add_prefix()
            print(to_plot)
            # plot_class_distribution_and_save(to_plot,N)
            


            return clients


        print("SPLITS!!!")
        # Output directory for saving plots
        output_directory = "client_plots"
        os.makedirs(output_directory, exist_ok=True)

        non_identical_clients = split_dataframe_to_non_identical_clients(self.df, N, alpha, num_clients, num_samples_per_client, p, output_directory)
        print(non_identical_clients)









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


    parser.add_argument("-alpha",
                    "--alpha",
                    help="alpha for dirchelet",
                    type=float,
                    required=True,
                    default=None)

    parser.add_argument("-beta",
                    "--beta",
                    help="beta fraction distribution",
                    type=float,
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
    alpha=float(args.alpha),
    beta=float(args.beta),
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
        alpha,
        beta,
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
