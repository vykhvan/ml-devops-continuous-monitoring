import pandas as pd
import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import os
import json
import subprocess

with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = os.path.join(config['input_folder_path'])
output_folder_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])

##################Check and read new data
#first, read ingestedfiles.txt
def run_ingestion():
    ingestedfiles = pd.read_csv(prod_deployment_path+"/ingestedfiles.txt", header=None)
    olddata = set(ingestedfiles.iloc[:,2].values)
    newdata = set(os.listdir(input_folder_path))
    if len(newdata.difference(olddata)) != 0:
        print("Run re-ingestion")
        ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)
    else:
        quit()

def run_training():
    with open(prod_deployment_path+"/latestscore.txt", "r") as file:
        latestscore = float(file.read())
    newestscore = scoring.score_model(output_folder_path+"/finaldata.csv", 
                                      prod_deployment_path)
    if newestscore < latestscore:
        print("Re-training")
        training.train_model()
    else:
        print("No model drift")
        quit()

def run_scoring():
    print("Run scoring")
    os.system("python scoring.py")

def run_deployment():
    print("Run deployment")
    deployment.store_model_into_pickle("trainedmodel.pkl")

def run_diagnostics():
    print("Run diagnostics")
    os.system("python diagnostics.py")

def run_reporting():
    print("Run reporting")
    os.system("python reporting.py")

def run_apicalls():
    print("Run calling API")
    os.system("python apicalls.py")

if __name__ == "__main__":
    run_ingestion()
    run_training()
    run_scoring()
    run_deployment()
    run_diagnostics()
    run_reporting()
    run_apicalls()
    