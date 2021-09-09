import subprocess
import os
import json
#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"

with open('config.json','r') as f:
    config = json.load(f) 
output_model_path = os.path.join(config['output_model_path']) 

#Call each API endpoint and store the responses
response1 = subprocess.run(['curl', URL+'/prediction?datapath=testdata/testdata.csv'], capture_output=True).stdout.decode('utf-8')
response2 = subprocess.run(['curl', URL+'/scoring'], capture_output=True).stdout.decode('utf-8')
response3 = subprocess.run(['curl', URL+'/summarystats'], capture_output=True).stdout.decode('utf-8')
response4 = subprocess.run(['curl', URL+'/diagnostics'], capture_output=True).stdout.decode('utf-8')

#combine all API responses
responses = response1 + '\n' + response2 + '\n' + response3 + '\n' + response4

#write the responses to your workspace
with open(output_model_path+"/apireturns.txt", "w") as file:
    file.write(responses)


