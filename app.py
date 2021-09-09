from flask import Flask, session, jsonify, request
import pandas as pd
import numpy as np
import pickle
import json
import os
import diagnostics
import scoring



######################Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prediction_model = os.path.join(config['prod_deployment_path'])
test_data_path = os.path.join(config['test_data_path'])

def readpandas(filename):
    thedata=pd.read_csv(filename)
    return thedata

#######################Prediction Endpoint
@app.route("/prediction", methods=['GET', 'OPTIONS'])
def get_predict():        
    #call the prediction function you created in Step 3
    datapath = request.args.get("datapath")
    dataframe = readpandas(datapath)
    prediction = diagnostics.model_predictions(dataframe)
    return str(prediction)

#######################Scoring Endpoint
@app.route("/scoring", methods=['GET','OPTIONS'])
def get_scoring():        
    score = scoring.score_model(test_data_path+"/testdata.csv", prediction_model)
    return str(score)

#######################Summary Statistics Endpoint
@app.route("/summarystats", methods=['GET','OPTIONS'])
def get_summarystats():        
    summary_statistics = diagnostics.dataframe_summary()
    return str(summary_statistics)

#######################Diagnostics Endpoint
@app.route("/diagnostics", methods=['GET','OPTIONS'])
def get_diagnosis():
    integrity_check = diagnostics.data_integrity_check()
    timing_check = diagnostics.execution_time()
    packages_check = diagnostics.outdated_packages_list()
    return str(integrity_check) + '\n' + str(timing_check) + '\n' + str(packages_check) + '\n'

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
