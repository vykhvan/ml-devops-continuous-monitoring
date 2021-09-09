# Project: a dynamic risk assessment system

1. Data ingestion. Automatically check a database for new data that can be used for model training. 
Compile all training data to a training dataset and save it to persistent storage. 
Write metrics related to the completed data ingestion tasks to persistent storage.

2. Training, scoring, and deploying. Write scripts that train an ML model that predicts attrition risk, and score the model. 
Write the model and the scoring metrics to persistent storage.

3. Diagnostics. Determine and save summary statistics related to a dataset. Time the performance of model training and scoring scripts. 
Check for dependency changes and package updates.

4. Reporting. Automatically generate plots and documents that report on model metrics. Provide an API endpoint that can return model predictions and metrics.

5. Process Automation. Create a script and cron job that automatically run all previous steps at regular intervals.
