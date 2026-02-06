
# P5 + T3 + ((P1 or P2 or P4) + (T2 + Gausian noise)
# P6 + T4 + (P1 or (P2 or P4) + (T2 + Gausian noise)
# P3 + T1 + (P1 or P2 or (P4) + (T2 + Gausian noise)

# Stack all!

# Find the best combination of P1,P2 and P4


# The dataset. Will contain 2 vuib sensors +1 temp sensor + t2 with gaussian noise!
# This will be repeated 3 so I will habe more data !

# Train in centralized model! Using MLP

# Get a good model!

# Perform federated on the basis we have 3 motor engine setups with different sensors palcement with 5 classses each





import pandas as pd


DIRECTORY='/home/jose/FL_AM_Defect-Detection/'



# df=pd.read_csv(DIRECTORY+'fl_testbed/version2/data/initial/original_combined_offset_misalignment.csv',index_col=None)#[lists]
df=pd.read_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment.csv',index_col=None)#[lists]

print(df.columns)

list1=[

'wf_start_time',
'Probe_1_AxialDirection_CrestFactor_g~g',
'Probe_1_AxialDirection_DerivedPeak_g',
'Probe_1_AxialDirection_Peak~Peak_g', 
'Probe_1_AxialDirection_RMS_g',
'Probe_1_AxialDirection_TruePeak_g',
'Probe_1_AxialDirection_HighFrequency_grms',
'Probe_1_AxialDirection_Kurtosis_g~g',
'Probe_5_RadialHorizontal_CrestFactor_g~g',
'Probe_5_RadialHorizontal_DerivedPeak_g',
'Probe_5_RadialHorizontal_Peak~Peak_g',
'Probe_5_RadialHorizontal_RMS_g', 
'Probe_5_RadialHorizontal_TruePeak_g',
'Probe_5_RadialHorizontal_HighFrequency_grms',
'Probe_5_RadialHorizontal_Kurtosis_g~g',

'Thermocouple 3_Value',



'status'
       
       ]

df_list1=df[list1]

df_list1.rename(columns={

    'Probe_1_AxialDirection_CrestFactor_g~g'        :'S1_CrestFactor_g~g',
    'Probe_1_AxialDirection_DerivedPeak_g'          :'S1_DerivedPeak_g',
    'Probe_1_AxialDirection_Peak~Peak_g'            :'S1_Peak~Peak_g', 
    'Probe_1_AxialDirection_RMS_g'                  :'S1_RMS_g',
    'Probe_1_AxialDirection_TruePeak_g'             :'S1_TruePeak_g',
    'Probe_1_AxialDirection_HighFrequency_grms'     :'S1_HighFrequency_grms',
    'Probe_1_AxialDirection_Kurtosis_g~g'           :'S1_Kurtosis_g~g',
    'Probe_5_RadialHorizontal_CrestFactor_g~g'      :'S2_CrestFactor_g~g',
    'Probe_5_RadialHorizontal_DerivedPeak_g'        :'S2_DerivedPeak_g',
    'Probe_5_RadialHorizontal_Peak~Peak_g'          :'S2_Peak~Peak_g',
    'Probe_5_RadialHorizontal_RMS_g'                :'S2_RMS_g', 
    'Probe_5_RadialHorizontal_TruePeak_g'           :'S2_TruePeak_g',
    'Probe_5_RadialHorizontal_HighFrequency_grms'   :'S2_HighFrequency_grms',
    'Probe_5_RadialHorizontal_Kurtosis_g~g'         :'S2_Kurtosis_g~g',
    'Thermocouple 3_Value'                          :'S1_temp',
    

},inplace=True)


list2=[
'wf_start_time',
'Probe_2_RadialVertical_CrestFactor_g~g',
'Probe_2_RadialVertical_DerivedPeak_g',
'Probe_2_RadialVertical_Peak~Peak_g', 
'Probe_2_RadialVertical_RMS_g',
'Probe_2_RadialVertical_TruePeak_g',
'Probe_2_RadialVertical_HighFrequency_grms',
'Probe_2_RadialVertical_Kurtosis_g~g',
'Probe_6_BearingRadial_CrestFactor_g~g',
'Probe_6_BearingRadial_DerivedPeak_g',
'Probe_6_BearingRadial_Peak~Peak_g', 
'Probe_6_BearingRadial_RMS_g',
'Probe_6_BearingRadial_TruePeak_g',
'Probe_6_BearingRadial_HighFrequency_grms',
'Probe_6_BearingRadial_Kurtosis_g~g',
'Thermocouple 4_Value',
'status'

]



df_list2=df[list2]

df_list2.rename(columns={

'Probe_6_BearingRadial_CrestFactor_g~g'   :'S1_CrestFactor_g~g',
'Probe_6_BearingRadial_DerivedPeak_g'   :'S1_DerivedPeak_g',
'Probe_6_BearingRadial_Peak~Peak_g'   :'S1_Peak~Peak_g', 
'Probe_6_BearingRadial_RMS_g'   :'S1_RMS_g',
'Probe_6_BearingRadial_TruePeak_g'   :'S1_TruePeak_g',
'Probe_6_BearingRadial_HighFrequency_grms'   :'S1_HighFrequency_grms',
'Probe_6_BearingRadial_Kurtosis_g~g'   :'S1_Kurtosis_g~g',
'Probe_2_RadialVertical_CrestFactor_g~g'   :'S2_CrestFactor_g~g',
'Probe_2_RadialVertical_DerivedPeak_g'   :'S2_DerivedPeak_g',
'Probe_2_RadialVertical_Peak~Peak_g'   :'S2_Peak~Peak_g',
'Probe_2_RadialVertical_RMS_g'   :'S2_RMS_g', 
'Probe_2_RadialVertical_TruePeak_g'   :'S2_TruePeak_g',
'Probe_2_RadialVertical_HighFrequency_grms'   :'S2_HighFrequency_grms',
'Probe_2_RadialVertical_Kurtosis_g~g'   :'S2_Kurtosis_g~g',
'Thermocouple 4_Value'   :'S1_temp',

},inplace=True)


list3=[
'wf_start_time',


'Probe_3_RadialHorizontal_CrestFactor_g~g',
'Probe_3_RadialHorizontal_DerivedPeak_g',
'Probe_3_RadialHorizontal_Peak~Peak_g',
'Probe_3_RadialHorizontal_RMS_g',
'Probe_3_RadialHorizontal_TruePeak_g',
'Probe_3_RadialHorizontal_HighFrequency_grms',
'Probe_3_RadialHorizontal_Kurtosis_g~g',
'Probe_4_RadialVertical_CrestFactor_g~g',
'Probe_4_RadialVertical_DerivedPeak_g',
'Probe_4_RadialVertical_Peak~Peak_g', 
'Probe_4_RadialVertical_RMS_g',
'Probe_4_RadialVertical_TruePeak_g',
'Probe_4_RadialVertical_HighFrequency_grms',
'Probe_4_RadialVertical_Kurtosis_g~g',

'Thermocouple 1_Value',

'status'

]

df_list3=df[list3]

df_list3.rename(columns={

'Probe_3_RadialHorizontal_CrestFactor_g~g':'S1_CrestFactor_g~g',
'Probe_3_RadialHorizontal_DerivedPeak_g':'S1_DerivedPeak_g',
'Probe_3_RadialHorizontal_Peak~Peak_g':'S1_Peak~Peak_g', 
'Probe_3_RadialHorizontal_RMS_g':'S1_RMS_g',
'Probe_3_RadialHorizontal_TruePeak_g':'S1_TruePeak_g',
'Probe_3_RadialHorizontal_HighFrequency_grms':'S1_HighFrequency_grms',
'Probe_3_RadialHorizontal_Kurtosis_g~g':'S1_Kurtosis_g~g',
'Probe_4_RadialVertical_CrestFactor_g~g':'S2_CrestFactor_g~g',
'Probe_4_RadialVertical_DerivedPeak_g':'S2_DerivedPeak_g',
'Probe_4_RadialVertical_Peak~Peak_g':'S2_Peak~Peak_g',
'Probe_4_RadialVertical_RMS_g':'S2_RMS_g', 
'Probe_4_RadialVertical_TruePeak_g':'S2_TruePeak_g',
'Probe_4_RadialVertical_HighFrequency_grms':'S2_HighFrequency_grms',
'Probe_4_RadialVertical_Kurtosis_g~g':'S2_Kurtosis_g~g',
'Thermocouple 1_Value':'S1_temp'},inplace=True)



print("DF1 ",df_list1.index.min, " : ",df_list1.index.max)
print("DF2 ",df_list2.index.min, " : ",df_list2.index.max)
print("DF3 ",df_list3.index.min, " : ",df_list3.index.max)
df=pd.concat([df_list1, df_list2, df_list3])
print(df.status.head())
print("HERE")
print(df.columns)

#WORKS FOR THE NOTEBOOKS
df.to_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment.csv')
del df



##ADITIONALLY OF CREATED THE "GROUPED" COMBINED CSV --> THIS IS MAINLY USED FOR THE NOTEBOOK !!! 
##THE SCRIPT WILL CREATE 3 SUBCSVs FILES SO THE IMPLEMENTATION OF SEPARATING IN MOTOR 1 2 AND 3 WILL BE DONE BY 
##THIS SCRIPT!



csv_file =DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment.csv'
df_temp = pd.read_csv(csv_file, chunksize=50000) 
df = pd.concat(df_temp, ignore_index=True)


#FOR EACH MOTOR "GROUP!!"

print(df[df.index==0])


#238722
#DO NOT FORGET SORT VALUES
df1=df.loc[0:238721].sort_values(by='wf_start_time').reset_index()
df2=df.loc[0+238722:0+238722+238722-1].sort_values(by='wf_start_time').reset_index()
df3=df.loc[0+238722+238722:].sort_values(by='wf_start_time').reset_index()



lists=[]
for df in [df1,df2,df3]:
    # Let's find the youngest & oldest timestamp

    df['wf_start_time'] = pd.to_datetime(df['wf_start_time']) # make sure it is datetime

    youngest = min(df.wf_start_time)
    oldest = max(df.wf_start_time)
    print(youngest)
    print(oldest)
    span = oldest - youngest
    print(span)
    print(span.total_seconds())

    ## Using Oldest - current to determine the RUL
    df['rul'] = df['wf_start_time'].apply(lambda x: (oldest - x).total_seconds())
    lists.append(df)




df1=pd.concat(lists[0:1],ignore_index=True)
print(df1.shape)
df1 = df1[df1.columns.drop(list(df1.filter(regex='Unnamed')))]


df2=pd.concat(lists[1:2],ignore_index=True)
print(df2.shape)
df2 = df2[df2.columns.drop(list(df2.filter(regex='Unnamed')))]


df3=pd.concat(lists[2:],ignore_index=True)
print(df3.shape)
df3 = df3[df3.columns.drop(list(df3.filter(regex='Unnamed')))]


df=pd.concat([df1,df2,df3])

df.to_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment_ALL.csv')

# df1.to_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment_M1.csv')

# df2.to_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment_M2.csv')

# df3.to_csv(DIRECTORY+'fl_testbed/version2/data/initial/combined_offset_misalignment_M3.csv')




