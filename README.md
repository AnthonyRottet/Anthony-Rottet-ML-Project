#Heart Disease Classification Project // Anthony Rottet 

## Installation & Usage
Please refer to the requirements.txt  

## Information
The GitHub repository is organized into three main folders: 
--- data --> it contains both the raw and processed data, there is nothing to execute 
--- results --> Contains the outputs of the analysis, including plots and tables. Here, you can find the individual 
    model results, the correlation matrix, and final performance summaries. Nothing to execute but you can have look if  
    you'd like to see how each model performed with the test set. 



--- src --> The source folder is divided into specialized sub-folders and .py scripts 
-Download.py / Data Cleaning.py / EDA.py --> These have already been executed to generate the final dataset. 
 The clean_data.csv is already included in the repository, but you can have a look to see the process. 
- Mutual Information.py --> This .py file executes the Mutual Information measurement. 
- Analyse --> Contains scripts used for the main results table, the KNN optimization plot, and a univariate logit 
  plotter used only during early exploratory phases but not in the project.   
- Model --> it contains the scripts for all the models with their final hyperparameter.
- Model Evaluation --> Scripts which evaluates the performance of each model.
- Grid Search --> Scripts which contains the Grid Search for each model.


-Main.py --> Regarding the Main.py, I decided that it will "only" execute each model and provide the optimized 
 hyperparameters, precision, recall, F1-score, support, AUC score, specificity, confusion matrix followed by a final
 comparative summary. I decided not to add the Download, Data cleaning, EDA and Mutual Information phase since their 
 outputs can easily be found in the results folder as described earlier. That is why the .main may be a bit short
 I hope this simplified workflow was the way to go ! :-D 

Have fun reading ! 



