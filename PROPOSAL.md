The goal of this project is to develop and compare several supervised machine learning models including: Multivariate 
Logistic Regression, Random Forest, K-Nearest Neighbors, Support Vector Machine and XGBoost. The dataset is about heart 
disease and is derived from a Cleveland Heart Disease Database, it can be found there : 
https://www.kaggle.com/datasets/neurocipher/heartdisease. In order to carefully select which variables will be 
used, a Mutual Information measurement will be conducted. In addition, I will develop a script for each model in order 
to find the performance of each model and then compare them, using the scikit-learn library. Then for each model, a 
Grid Search with 5-fold cross-validation will be performed in order to find the best hyperparameters and implement them 
in the classifiers. Ultimately, the different models will be compared utilizing the AUC score, Recall and F1-score and
Accuracy. I will also analyse the probability of False Negatives for each model as it is a very important metric in the 
clinical field