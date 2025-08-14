# Dragon Real Estate Investment Analysis (CodeWithHarry)
# This script analyzes real estate investment opportunities using machine learning techniques.

# Project 1: Hands on Machine Learning with Python
# Problem Teardown:
# We are given a dataset of houses  prices with features like no of bathrooms, bedrooms, etc.
# Our task is create a model which will predict the price for any new house based on its features.
# While learning about ML it is best to actually work with a real dataset not just for artificial datasets.
# There are hundreds of open datasets to choose from, we have already talked about same ML course.

# Getting Started:
# The first que. taskin should ask Mr.josheph is what the bussiness objective and end goal? how will dragon real estate benefit from the model?
# Mr.josheph tells taskin that dragon real estate will use this model to predict the price of new houses based on area and will invest in the area if its undervalued.
# Next que. taskin should ask mr. is how does the current solution look like ? ans is - manual experts who analyze the features.
# the predicition made by so called experts are not accurate(error rate is 25%) which is why they counting on taskin. 

# # Finding the type of model to built
# Supervised,unsupervised or reinforcement learning? => supervised learning
# classification or regression? => regression
# # In machine learning, "batch learning" means the model is trained on the entire dataset at once, rather than updating continuously as new data arrives. The model learns from a fixed batch of data, and if new data becomes available, the model must be retrained from scratch or with the new batch. 
# # This is different from "online learning," where the model updates incrementally as each new data point arrives. Batch learning is suitable when you have all your data available upfront and don't need real-time updates.
# batch learning or online learning? => online learning

# Selecting performance Measure:
# A typeical performance measure for regression tasks is the Root Mean Squared Error (RMSE).
# RMSE is generall the preferred performace measure for regression tasks, so we choose it for this particular problem we are solving for dragon real estate.
# Other performance measures include Mean Absolute Error (MAE), manhattan norm, etc but we will use RMSE for this problem.
# EX: It calculates the difference between each predicted value and the actual value (e.g., predicted 3 cr, actual 7 cr, error = -4 cr; predicted 7 cr, actual 3 cr, error = +4 cr).
# It squares these errors (so both -4 and +4 become 16), which means large errors have a bigger impact.
# It averages the squared errors and then takes the square root, giving you a single number that represents the typical prediction error in the same units as your target (e.g., crores).

# Checking the assumptions:
# it is very important to check the assumptions he might have made and correct them before launching the ML system.
# ex, he should make sure that the team needs the price and not the category like expensive,cheap etc.
# if latter is the case, formulating the problem as a regression problem will be counted as big mistake.
# taskin talked to dragon real estate teams and clarified their requirements.

# All set for coding now...
# the light are green, taskin can start coding now.
# taskin will use a VS code.
# the VS code is already set up with the necessary libraries installed.

# CRIM (Crime rate): If CRIM = 0.5, it means the town is very safe. If CRIM = 10, it’s less safe.
# ZN (Land for big houses): If ZN = 30, then 30% of the land is for big houses (like bungalows).
# INDUS (Industry area): If INDUS = 20, then 20% of the town is factories/offices, not homes or shops.
# CHAS (Near river): If CHAS = 1, the house is next to the Charles River. If CHAS = 0, it’s not.
# NOX (Air pollution): If NOX = 0.4, the air is cleaner. If NOX = 0.8, there’s more pollution.
# RM (Rooms per house): If RM = 3, the house is small (3 rooms). If RM = 7, it’s big (7 rooms).
# AGE (Old houses): If AGE = 80, then 80% of houses were built before 1940 (very old area). If AGE = 20, most houses are newer.
# DIS (Distance to jobs): If DIS = 2, the house is close to job centers. If DIS = 10, it’s far away.
# RAD (Highway access): How easy it is to access highways. Higher means better access. If RAD = 1, it’s hard to reach highways. If RAD = 24, it’s very easy.
# TAX (Property tax): If TAX = 200, taxes are low. If TAX = 700, taxes are high.
# PTRATIO (Students per teacher): If PTRATIO = 12, classes are small (good). If PTRATIO = 22, classes are big (less attention).
# B (Black population): If B = 400, there’s some diversity. If B = 390, less diversity.
# LSTAT (Low-income residents): If LSTAT = 5, only 5% are low-income (wealthy area). If LSTAT = 30, 30% are low-income (less wealthy).
# MEDV (House price): If MEDV = 20, the median house price is $20,000. If MEDV = 50, it’s $50,000.

# Import necessary libraries
import pandas as pd  # For handling data as a DataFrame
import matplotlib.pyplot as plt  # For plotting graphs
import numpy as np  # For numerical operations

housing = pd.read_csv('data.csv',usecols=range(14))  # Load the dataset
print(housing.head())  # Display the first few rows of the dataset
print(housing.info())  # Display information about the dataset
print(housing['CHAS'].value_counts())  # Display the 'CHAS' column to check if it contains only 0s and 1s
print(housing.describe())  # Get a statistical summary of the dataset

# bins=50: This means each histogram will divide the data into 50 intervals (bars). More bins give you a more detailed view of how the data is distributed.
# figsize=(20, 15): This sets the size of the whole figure (width=20, height=15 inches). A bigger figure makes all the plots easier to see, especially when you have many feature
housing.hist(bins=50,figsize=(20,15))  # Create histograms for each feature in the dataset
# plt.show()
# CRIM (Crime rate): Usually, most towns have low crime rates, so you’ll see a tall bar on the left and short bars on the right (right-skewed).
# ZN (Land for big houses): Most values are zero, so you’ll see a tall bar at zero and maybe a few bars at higher values.
# INDUS (Industry area): You’ll see how many towns have more or less industry. If most bars are on the left, most towns have little industry.
# CHAS (Near river): Only two bars: one for 0 (not near river, usually taller), one for 1 (near river, usually shorter).
# NOX (Air pollution): If most bars are on the left, most towns have cleaner air.
# RM (Rooms per house): Most houses have 5–7 rooms, so the tallest bars are in that range.
# AGE (Old houses): If the bars are spread out, there’s a mix of old and new houses.
# DIS (Distance to jobs): If most bars are on the left, most houses are close to job centers.
# RAD (Highway access): You may see a few tall bars at certain values, showing how many towns have good or poor highway access.
# TAX (Property tax): Tall bars at lower values mean most towns have lower taxes.
# PTRATIO (Students per teacher): If the bars are bunched together, most towns have similar class sizes.
# B (Black population): The distribution shows diversity in towns.
# LSTAT (Low-income residents): Tall bars at lower values mean most towns are wealthier.
# MEDV (House price): Tallest bar at $20,000 means most houses are around that price. If there are bars at higher prices, some houses are expensive.


# Only for learning purpose beacuse python it give built in code like train_test_models:
# Training and Testing Data:
# def split_train_test(data,test_ratio):
#     np.random.seed(42) # for one type of datasets
#     # np.random.permutation(len(data)) generates a random permutation of the indices of the dataset.
#     # If you have len(data) = 1000, it returns a shuffled array of numbers from 0 to 999.
#     # This shuffling makes sure the split is random.
#     shuffled = np.random.permutation(len(data))  # Shuffle the data randomly
#     # If you set test_ratio = 0.2 (i.e., 20%), and you have len(data) = 1000, then:
#     # test_set_size = int(1000 * 0.2) = 200
#     test_set_size = int(len(data)  * test_ratio)  # This calculates how many rows should go into your test set. If you want 20% for testing and you have 100 rows, it gives you 20.
#     # test_indices gets the first 200 shuffled indices, which become the test set.
#     # train_indices gets the remaining 800 indices, which become the training set.
#     test_indices = shuffled[:test_set_size] # Select the first 'test_set_size' indices for the test set
#     train_indices = shuffled[test_set_size:]  # The rest are for the training set
#     # .iloc[] is a Pandas function used to access data by index location (integer-based indexing).
#     # So, you are using the randomly selected indices to retrieve rows from the DataFrame.
#     return data.iloc[train_indices], data.iloc[test_indices]  # Return the training and test sets as DataFrames

# train_set , test_set = split_train_test(housing,0.2)
# # print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)} \nRows in test set: {len(test_set)}\n")


# jo ahi CHAS ni value ni andar 471 = 0 and 35 = 1 chh to e ek type ma na jata rhe like banne ma data half half ava pde like koi alien earth pr ave and just amarican people notice kre to wrong kehvay emne badha vise khbr hovi pde.
# 🧠 Why It’s Useful
# Let’s say:
# You have a dataset of medical test results
# Each row belongs to a patient
# Each patient can have multiple test results (i.e., multiple rows per group)
# Your target is whether the patient has a disease (label: 0 or 1)
# You don’t want:
# Rows from the same patient to appear in both train and test (leakage)
# But you also want each fold to have the same proportion of diseased vs. healthy patients (stratification)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']): # Verifies if the split preserved the ratio of 0s and 1s in CHAS.
    strat_train_set = housing.loc[train_index] # Selects actual rows from the DataFrame using those indices.
    strat_test_set = housing.loc[test_index]

# print(strat_test_set)
# print(strat_test_set.describe())
# print(strat_test_set.info())
# print(strat_test_set['CHAS'].value_counts())
# print(strat_train_set['CHAS'].value_counts())

housing = strat_train_set.copy()

# Looking for Correlations
# Correlation refers to a statistical relationship or connection between two or more variables. It tells you whether — and how strongly — changes in one variable are associated with changes in another.
# r = ∑(xi-(mean of x))(yi−(mean of y))/ squere root of (∑(xi−(mean of x))) squere root of (∑(yi−(mean of y)))
# or 
# r= Cov(X,Y)/σX.σY
# Cov(X, Y) = covariance of X and Y
# 𝜎𝑋σ𝑌 = standard deviations of X and Y
corr_metrix = housing.corr()
a = corr_metrix['MEDV'].sort_values(ascending=False) 
# print(a) # here if MEDV then after RM beacuse MEDV increse (1) so RM(room) are increase. it is correlation with eachothers.add()
# plt.show()

import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
# scatter_matrix() creates the figure, but does not automatically display it in some environments (like scripts or some IDEs).
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()  
# Here Purpose of Comparing These Pairs:
# 1. MEDV vs. RM
# You're checking if more rooms = higher house price (MEDV).
# Usually shows a positive correlation — more rooms, higher value.
# 2. MEDV vs. LSTAT
# You're seeing if a higher % of low-income population = lower house prices.
# Usually shows a negative correlation — more LSTAT, lower MEDV.
# 3. MEDV vs. ZN
# You're checking if homes in low-density, residential zones are more valuable.
# May or may not show a clear trend — depends on your data.

housing.plot(kind="scatter", x="RM", y="MEDV", alpha=1)
# plt.show()
# # ahi hu outlier nikadi apis and ML pattern pn saru mdse so prediction pan sachu mdse jem k RM = 5 pase MEDV = 50 chh and RM = 9 pase rpan MEDV = 50 chh je outlier chh 

# Trying Out Attribute Combinations:
housing["TAXRM"] = housing['TAX']/housing['RM']
print(housing["TAXRM"])
# Correlations
corr_metrix = housing.corr()
a = corr_metrix['MEDV'].sort_values(ascending=False) 
print(a) # ahi TAXRM name nu new varible add thai jse for TAXRM
# plot it
# housing.plot(kind="scatter",x="TAXRM",y="MEDV",alpha = 0.8)

housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()

# Missing Attribute -> here i miss data in RM columan so we check how it work here 3 option is best for set value still we try all.
# To take care of missing attributes, you have three options:
# Option 1. Get rid of the missing data points.
# Oprion 2. Get rid the whole attributes.
# Option 3.Set the value to some value (0, mean or median)

# a = housing.dropna(subset=["RM"]) # Option-1
# print(a.shape) # the original housing dataframe will remain unchanged

# b = housing.drop("RM",axis=1).shape # Option 2
# print(b) 

median = housing["RM"].median() # Option 3
print(median)
c = housing["RM"].fillna(median)
print(c)
print(housing.describe())
# ✅ What is SimpleImputer?
# SimpleImputer is a tool from Scikit-Learn that automatically fills in (imputes) missing values in a dataset using a strategy you choose — like: mean,median,most_frequent,constant
from sklearn.impute import SimpleImputer 
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)
# Here get value for all column like RM,MEDV etc 
d = imputer.statistics_ 
print(d)
X = imputer.transform(housing)
housing_tr = pd.DataFrame(X,columns=housing.columns)
print(housing_tr.describe())

# Scikit-learn Design 
# Primarily , three type of object:
# 1. Estimators - it estimates some parameters based on dataset. eg.imputer it has fit method and transform method. fit method - Fits the dataset and calculates iternal perameter.
# These are any objects that learn from data.
# They usually have:
# .fit() method → to learn from the data
# Sometimes also .transform() or .predict()
# 2. Transformers - transform takes input and returns output based on the learing from fit(). it also has a convenience function called fit_transform() which fits and then transforms.
# These take what was learned in .fit() and apply it to the data using .transform().
# They can also use .fit_transform() to do both steps in one.
# ✅ Transformers are a type of estimator that has .transform() — like StandardScaler, SimpleImputer, etc.
# 3. Predictors - LinearRegression model is  an example of predictor . fit() and predict() are two common function.it also gives score function which will evalute the predicitons.
# These are machine learning models.
# They take input features, learn from them, and make predictions.
# They use:
# .fit(X, y) → to train the model
# .predict(X) → to make predictions
# .score(X, y) → to evaluate performance

# Feature Scaling
# two types of feature scaling methods:
# 1. Min-Max scaling (normalization)
#     (value - min)/(max-min)
#     sklearn provides a class called MinMaxScaler for this
# 2. Standardizatiom (Z-score)
#     (value-mean)/std
#     sklearn provides a class called StandardScaler for this 

# Create Pipline
# pipline:A pipeline in machine learning is a step-by-step workflow that automates the data preprocessing and model training process.
# Think of it like a conveyor belt in a factory:
# Raw materials (your raw data) go in,
# They pass through machines (data cleaning, feature selection, model training),
# And you get a finished product (a trained model or prediction).

# 📦 What’s Inside a Pipeline?
# A pipeline usually includes these steps:
# Data Preprocessing
# Handling missing values (SimpleImputer)
# Scaling (StandardScaler, MinMaxScaler)
# Encoding categorical variables (OneHotEncoder)
# Feature Selection or Transformation (optional)
# PCA, PolynomialFeatures, etc.
# Model Training
# A model like LinearRegression, RandomForestClassifier, etc.
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer',SimpleImputer(strategy = "median")),   # .... add as many you want in your pipline
    ('std_scaler',StandardScaler())
])
housing_num_tr = my_pipeline.fit_transform(housing)
print(housing_num_tr)
print(housing_num_tr.shape)

# selecting a desired model for dragon real estates
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
#  Create the model
#  It's a good baseline model for regression problems.  
#  Assumes a straight-line relationship between features and target (price).
model = LinearRegression()
# model = DecisionTreeRegressor() # after we see overfitting
# housing_num_tr: your training data (numerical features of houses).
# housing_labels: the actual house prices (target values).
# .fit() → learns the relationship between features and target (creates the model's parameters)
model.fit(housing_num_tr,housing_labels)
# Prepare some sample data to test predictions manually
# .iloc[:5] → select the first 5 rows of the housing DataFrame by position (integer index)
# We take a small subset just to see how the model predicts for known values
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
# Preprocess the sample data
# We must transform the new data using the SAME pipeline used for training:
# Fills missing values
# Scales the features
# This ensures our input format matches what the model was trained on
# If we skip this step, the model may give wrong results or throw errors
prepared_data = my_pipeline.transform(some_data)
#  Make predictions:predict() → takes the prepared features and outputs predicted house prices
a  = model.predict(prepared_data)
print(a)
print(some_labels)

#Evaluating the model
from sklearn.metrics import mean_squared_error
# Predict prices for the entire training set
# This helps us measure how well the model fits the training data
housing_predictions = model.predict(housing_num_tr)
# mean_squared_error → average of squared differences between actual and predicted prices
# Squaring ensures positive errors and penalizes large mistakes more
mse = mean_squared_error(housing_labels,housing_predictions)
# Taking the square root brings error back to original units (e.g., lakhs)
rmse = np.sqrt(mse)
print(mse) # overfitting here in desicion tree

# better evalution technique - Cross validation (here remove overfitting)
# Performs cross-validation to test your model’s performance more reliably.
# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
# a) model : This is your ML model (e.g., LinearRegression() or any regression model you created earlier). Why pass it? cross_val_score will train and evaluate this model on different folds.
# b) housing_num_tr:Your features (X) after preprocessing/transformation (e.g., scaling, encoding, etc.).
# c) housing_labels:Your target values (y), e.g., house prices.
# d) scoring="neg_mean_squared_error":In regression, we often measure error using MSE (Mean Squared Error).
# Problem: In scikit-learn, higher scores are considered better. But MSE is a "loss" metric (lower is better), so sklearn returns the negative of MSE.
# That’s why it’s "neg_mean_squared_error" — so the function still works with the “higher-is-better” logic. We later take -scores to get positive MSE.
# cv=10 means 10-fold cross-validation: Split dataset into 10 parts. Train on 9 parts, test on 1 part.
scores = cross_val_score(model, housing_num_tr,housing_labels,scoring="neg_mean_squared_error",cv=10)
# scores → array of negative MSE values from cross-validation. scores → converts them back to positive MSE values.
rmse_scores = np.sqrt(-scores)
print(rmse_scores)

def print_score(scores):
    print("scores:",scores)
    print("mean:",scores.mean())
    print("STD:",scores.std())

print(print_score(rmse_scores))

# Desicion tree
# mean: 4.013521050382086
# STD: 0.7468147793966334

# Linear Regression
# mean: 5.028914500287415
# STD: 1.0611970088620446

from joblib import dump,load
# dump(model,'Dragon.joblib')
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Dragon.joblib")
dump(model, file_path)
print("Saved to:", file_path)


# 1. What is joblib?
# joblib is a Python library used for saving and loading Python objects to a file.
# It is especially good for large data objects (like NumPy arrays, Pandas DataFrames, or trained ML models) because it saves them in a compressed binary format that is fast to load.
# In ML, after we train a model, we don’t want to train it again every time. So, we save it to disk using joblib and later load it instantly.
# 📦 Think of it like putting your model into a zip file and storing it, so you can take it out later without retraining.

# 2. What is dump()?
# Meaning: Save the Python object (model) into a file.
# model → your trained ML model object.
# 'Dragon.joblib' → file name where the model will be saved.

# Testing the model om test data
X_test = strat_test_set.drop("MEDV",axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_prediction = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test,final_prediction)
final_rmse = np.sqrt(final_mse)
print(final_rmse)
print(final_prediction,list(Y_test))