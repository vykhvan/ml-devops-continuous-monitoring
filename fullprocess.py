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
print("Check new data")
ingestedfiles = pd.read_csv(prod_deployment_path+"/ingestedfiles.txt", header=None)
olddata = set(ingestedfiles.iloc[:,2].values)
#second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
newdata = set(os.listdir(input_folder_path))
if len(newdata.difference(olddata)) != 0:
    print("Run re-ingestion")
    ingestion.merge_multiple_dataframe(input_folder_path, output_folder_path)

##################Deciding whether to proceed, part 1
#if you found new data, you should proceed. otherwise, do end the process here
if len(newdata.difference(olddata)) == 0:
    print("No new data - end process")
    quit()

##################Checking for model drift
#check whether the score from the deployed model is different from the score from the model that uses the newest ingested data
print("Check model drift")
with open(prod_deployment_path+"/latestscore.txt", "r") as file:
    latestscore = float(file.read())
newestscore = scoring.score_model(output_folder_path+"/finaldata.csv", prod_deployment_path)
if newestscore <= latestscore:
    print("Re-training")
    training.train_model()
##################Deciding whether to proceed, part 2
#if you found model drift, you should proceed. otherwise, do end the process here
if newestscore > latestscore:
    print("No model drift")
    quit()


##################Re-deployment
#if you found evidence for model drift, re-run the deployment.py script
print("Re-deployment")
os.system("python scoring.py")
deployment.store_model_into_pickle("trainedmodel.pkl")
##################Diagnostics and reporting
#run diagnostics.py and reporting.py for the re-deployed model
os.system("python diagnostics.py")
os.system("python reporting.py")
os.system("python apicalls.py")


