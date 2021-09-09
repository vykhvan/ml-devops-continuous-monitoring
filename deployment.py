from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



##################Load config.json and correct path variable
with open('config.json','r') as f:
    config = json.load(f) 

output_folder_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path'])
####################function for deployment
def store_model_into_pickle(model):
    #copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    model_path = os.path.join(output_model_path, model)
    score_path = os.path.join(output_model_path, 'latestscore.txt')
    records_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    os.system(f"cp {model_path} {prod_deployment_path}")
    os.system(f"cp {score_path} {prod_deployment_path}")
    os.system(f"cp {records_path} {prod_deployment_path}")
    
  
if __name__ == "__main__":
    store_model_into_pickle("trainedmodel.pkl")
        
        

