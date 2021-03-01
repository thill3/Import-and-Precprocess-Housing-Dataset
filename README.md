# Import-and-Preprocess-Housing-Dataset

This code is intended to find, load, and preprocess the "housing.txt" dataset

Comments should help indicate the fuctions of the various pieces of code. 

Overall summary is below.

1) Function to fetch the .tgz file from its location online and download it.
2) Function to load the downloaded data file so that it can be processed.
3) Income numbers are divided into 5 categories
4) Data is split into train and test sets using a stratified shuffle based on the income category.
5) Divide the features (predictors) from the labels (outputs).
6) Divide the features into numerical features and categorical features.
7) Determine whether we have nulls so that we can impute values.
8) Create Pipelines to process the numerical and categorical data separately.
  a) Numerical data is imputed with the median and scaled.
  b) Categorical data is One Hot encoded.
9) The two separate pipelines are combined.
