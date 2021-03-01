# Import-and-Preprocess-Housing-Dataset

This code is intended to find, load, and preprocess the "housing.txt" dataset

Comments should help indicate the fuctios of the various pieces of code. Here is a summary.

Function to fetch the .tgz file from its location online and download it.
Function to load the downloaded data file so that it can be processed.
Income numbers are divided into 5 categories
Data is split into train and test sets using a stratified shuffle based on the income category.
Divide the features (predictors) from the labels (outputs).
Divide the features into numerical features and categorical features.
Determine whether we have nulls so that we can impute values.
Create Pipelines to process the numerical and categorical data separately.
  Numerical data is imputed with the median and scaled.
  Categorical data is One Hot encoded.
The two separate pipelines are combined.
