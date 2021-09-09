
import pandas as pd
import numpy as np
import timeit
import os
import json
import pickle
import subprocess

##################Load config.json and get environment variables
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

def readpandas(filename):
    thedata=pd.read_csv(filename)
    return thedata

##################Function to get model predictions
def model_predictions(dataframe):
    #read the deployed model and a test dataset, calculate predictions
    with open(prod_deployment_path+"/trainedmodel.pkl", "rb") as file:
        model = pickle.load(file)
    features = dataframe[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    predictions = model.predict(features)
    return predictions

##################Function to get summary statistics
def dataframe_summary():
    #calculate summary statistics here
    traindata = pd.read_csv(output_folder_path+"/finaldata.csv")
    columns = ["lastmonth_activity", "lastyear_activity", "number_of_employees"]
    summary_stat = {}
    for column in columns:
        mean = np.mean(traindata[column])
        median = np.median(traindata[column])
        std = np.std(traindata[column])
        summary_stat.update({column: {"mean": mean, "median": median, "std": std}})
    return summary_stat

##################Function to check data integrity
def data_integrity_check():
    traindata = pd.read_csv(output_folder_path+"/finaldata.csv")
    nas=list(traindata.isna().sum())
    napercents=[nas[i]/len(traindata.index) for i in range(len(nas))]
    integrity = {}
    for i in range(len(napercents)):
        integrity.update({traindata.columns[i]: napercents[i]})
    return integrity

##################Function to get timings
def execution_time():
    #calculate timing of training.py and ingestion.py
    exec_timing = {}
    for step in ["ingestion.py", "training.py"]:
        starttime = timeit.default_timer()
        os.system(f"python {step}")
        timing=timeit.default_timer() - starttime
        exec_timing.update({step: timing})
    return exec_timing

##################Function to check dependencies
def outdated_packages_list():
    installed = subprocess.check_output(['pip', 'list', '--outdated'])
    return installed.decode('utf-8')


if __name__ == '__main__':
    print(model_predictions(readpandas(test_data_path+"/testdata.csv")))
    print(dataframe_summary())
    print(data_integrity_check())
    print(execution_time())
    print(outdated_packages_list())





    
