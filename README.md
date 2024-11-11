Intrusion Detection System using Machine Learning

Project Overview

This project aims to build an Intrusion Detection System (IDS) using machine learning models to classify network traffic data as either normal or attack. It leverages various machine learning models such as Random Forest, CNN-LSTM, and XGBoost, and includes preprocessing techniques like scaling, encoding, and handling imbalanced data. The project uses the CICIDS 2018 dataset for traffic classification.

Key Features

Data Preprocessing: Cleans and prepares data by handling missing values, duplicates, and categorical variables.
Modeling: Implements multiple classification models, including:
Random Forest with class weighting for imbalanced data
A hybrid CNN-LSTM model for feature extraction
XGBoost for final classification
Hyperparameter Tuning: Uses RandomizedSearchCV for optimizing hyperparameters of models.
Evaluation: Evaluates models based on F1-score, accuracy, and confusion matrix.
Technologies Used
Python: The programming language used for implementing machine learning models.
Libraries:
Pandas: For data manipulation and analysis.
NumPy: For numerical computations.
Matplotlib/Seaborn: For data visualization.
Scikit-learn: For machine learning models, preprocessing, and evaluation.
Keras/TensorFlow: For deep learning models (CNN-LSTM).
XGBoost: For gradient boosting classification.
Joblib: For saving and loading machine learning models.
Data Description
The dataset used in this project is from the CICIDS 2018 dataset, which contains network traffic data captured in two days:

Thursday-15-02-2018
Friday-16-02-2018
The dataset includes features related to network traffic such as Flow ID, Protocol, Source IP, Destination IP, Bytes, Packets, and more, along with a target label indicating whether the traffic is normal or an attack.


Model Evaluation
Random Forest: The model is evaluated with a focus on handling imbalanced data using class weights.
CNN-LSTM: A hybrid model to extract complex features from data for improved classification.
XGBoost: Optimized using RandomizedSearchCV to find the best hyperparameters.
The models are evaluated using F1-score (weighted) and accuracy. A confusion matrix is also plotted to visualize the classification performance.

Results
The Random Forest model, when trained with class weights, yields strong F1 scores on both the training and test sets.
The CNN-LSTM model extracts features and improves classification accuracy before being passed to XGBoost.
The XGBoost model outperforms others with hyperparameter tuning and class balancing.