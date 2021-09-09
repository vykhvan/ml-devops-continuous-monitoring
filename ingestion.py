import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import csv

#############Load config.json and get input and output paths
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

#############Function for data ingestion
def merge_multiple_dataframe(input_pth, output_pth):
    """
    Merge multiple dataset
    Inputs
    ------
    input_pth : str
        Input folder path to data.
    output_pth : str
        Output folder path for merged data.
    Returns
    -------
    ingest_data: pandas.core.frame.DataFrame
        Ingest data.
    """
    #check for datasets, compile them together, and write to an output file
    finaldata = pd.DataFrame(columns=["corporation",
                                        "lastmonth_activity",
                                        "lastyear_activity",
                                        "number_of_employees",
                                        "exited"])

    directory_pth = os.path.join(os.getcwd(), input_pth)
    datalist = os.listdir(directory_pth)
    for filename in datalist:
        data_pth = os.path.join(directory_pth, filename)
        datatemp = pd.read_csv(data_pth)
        finaldata=finaldata.append(datatemp)
        ingesttime = str(datetime.now())
        row = [ingesttime, directory_pth, filename, len(datatemp)]
        with open(output_pth+"/ingestedfiles.txt", 'a') as file:
            writer = csv.writer(file)
            writer.writerow(row)    
      
    finaldata = finaldata.drop_duplicates()
    
    ingest_pth = os.path.join(output_pth, "finaldata.csv")
    finaldata.to_csv(ingest_pth, index=False)

        
if __name__ == '__main__':
    merge_multiple_dataframe(input_folder_path,
                             output_folder_path)
