import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import diagnostics



###############Load config.json and get path variables
with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
output_model_path = os.path.join(config['output_model_path']) 



##############Function for reporting
def score_model():
    #calculate a confusion matrix using the test data and the deployed model
    #write the confusion matrix to the workspace
    testdata = pd.read_csv(test_data_path+"/testdata.csv")
    test_labels = testdata['exited']
    preds_labels = diagnostics.model_predictions(testdata)
    conf_mat = confusion_matrix(test_labels, preds_labels)
    plt.figure(figsize=(5, 5))
    sns.heatmap(conf_mat)
    plt.savefig(output_model_path+"/confusionmatrix.png")



if __name__ == '__main__':
    score_model()
