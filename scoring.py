from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import json



#################Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 

#################Function for model scoring
def score_model(test_pth, model_pth):
    #this function should take a trained model, load test data, and calculate an F1 score for the model relative to the test data
    #it should write the result to the latestscore.txt file
    testdata = pd.read_csv(test_pth)
    with open(model_pth+"/trainedmodel.pkl", "rb") as file:
        model = pickle.load(file)
    features = testdata[["lastmonth_activity", "lastyear_activity", "number_of_employees"]]
    labels = testdata["exited"]
    predictions = model.predict(features)
    score = metrics.f1_score(labels, predictions)
    with open(model_pth + "/latestscore.txt", "w") as file:
        file.write(str(score))
    return score
        
if __name__ == "__main__":
    print(score_model(test_data_path+"/testdata.csv", output_model_path))

