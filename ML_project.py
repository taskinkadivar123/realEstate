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
from sklearn.model_selection import StratifiedGroupKFold

# Replace 'GROUP_COLUMN' with the actual column in your DataFrame
group_column = 'GROUP_COLUMN'  # <-- change this to the real column name

splitter = StratifiedGroupKFold(n_splits=2, shuffle=True)

for train_index, test_index in splitter.split(housing, housing['CHAS'], groups=housing[group_column]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    break  # take the first fold

print(strat_test_set)
