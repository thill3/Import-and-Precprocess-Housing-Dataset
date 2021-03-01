################################################################
#Analysis Setup
################################################################

#fetch the data
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

fetch_housing_data()

#load the data
import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()

#categorize the income feature
import numpy as np
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])

#Stratified split the data
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#set up the train data features
train_features = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set

#set up the train data labels
train_labels = strat_train_set["median_house_value"].copy()

#split the train features into numerical and categorical
train_features_num = train_features.drop("ocean_proximity", axis = 1)
train_features_cat = ["ocean_proximity"]

#how many nulls are there?
#print(housing_num.isnull().sum())
print(train_features.isnull().sum())
print(train_features.shape)
#So there are 158 nulls out of 16512 rows in the "total_bedrooms" column
#that'll be a job for the simple imputer

#let's prepare the intermediate pipelines
#to create the pipeline to process the data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder)
])

#More Pipeline
num_attribs = list(train_features_num)
cat_attribs = list(train_features_cat)

from sklearn.compose import ColumnTransformer
prepare_the_data = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OneHotEncoder(), cat_attribs),
    ])

#Create a dataset o prepared training data
housing_train_prepared = prepare_the_data.fit_transform(strat_train_set)
housing_train_prepared = pd.DataFrame(housing_train_prepared)
housing_train_prepared.shape
#16512x14