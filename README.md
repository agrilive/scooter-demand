# Predicting the number of active e-scooter users

**Problem statement**: Predict the total number of active e-scooter users with the given attributes 

**Approach**:
1. Data extraction
2. Exploratory data analysis 
3. End-to-end machine learning pipeline

## Running the files

1. Change directory to this folder
2. In the command prompt, type

```
sh run.sh
```
This would run the _run.sh_ file, which loads the installs the necessary packages under the _requirements.txt_ file and runs the _main.py_ Python script.

## (1) Data Extraction

The dataExtraction.py file uses SQL query to extract the dataset from the given connection details. The extracted data is then saved as _rental_data.csv_.

We need these variables from _rental_data_:
```
date:                   YYYY-MM-DD format
hr:                     0 to 23
weather:                condition for the hr
temperature:            average for the hr (Fahrenheit)
feels_like_temperature: average feeling for the hr (Fahrenheit)
relative_humidity:      average for the hr
windspeed:              average for the hr
psi:                    pollutant standard index (0 to 400)
guest_scooter:          no. of guest users for the hr
registered_scooter:     no. of registered users for the hr
```

## (2) Exploratory Data Analysis

The _eda.ipynb_ file shows the data visualization of the rental dataset. This file can be split into two parts:

1. Hypothesis generation and testing
2. Time series analysis

Both of these would help in identifying the variables that need to be pre-processed.

Notably, there were many outliers in the _guest_scooter_ and _registered_scooter_ dependent variables. We will be using a logarithmic transformation to deal with these outliers. 

## (3) End-to-End Machine Learning Pipeline

The _main.py_ file contains the entire machine learning pipeline from data extraction to feature engineering. This file will call the functions from other files: 

1. **dataExtraction.py** - Mentioned above in _(1) Data Extraction_
2. **featureEngineering.py** - Cleans the data and returns a _clean_data.csv_ file
3. **regression.py** - Fits and predicts the clean dataset to the various supervised learning regression models

### Model choices

3 regression models were used 

1. Ridge regression
2. Lasso regression
3. Random forest regression

An important point to note is that the guest users and registered users were trained on separate models. From the EDA, we find out that these two groups have different seasonality trend although they exhibit the same overall trend.

In addition, a new scoring method was created using make_scorer from sklearn.metrics. We used the Root Mean Squared Logarithmic Error to calculate the error since we are took the logarithmic transformation for the guest_scooter and registered_scooter variables.

### Model evaluation

The error metrics used is the **Root Mean Squared Logarithmic Error** (RMSLE). RMSLE incurs a larger penalty for the underestimation of the actual variable than its overestimation. This is especially useful for when the underestimation of the target variable is not acceptable but overestimation can be tolerated.

After using GridSearchCV to find the best parameters, the RandomForestRegressor gave the lowest RMSLE values. The scores are as follows:

#### (1) Random forest regressor
```
Random forest regression for guest_scooter
Best parameters: {'max_depth': 7, 'n_estimators': 300}
RMSLE score: 0.23545610486361496

Random forest regression for registered_scooter
Best parameters: {'max_depth': 8, 'n_estimators': 500}
RMSLE score: 0.12729247135692237
```

#### (2) Ridge regression
```
Ridge regression for guest_scooter
Best parameters: {'alpha': 0.01, 'max_iter': 3000}
RMSLE score: 0.27159622182418625

Ridge regression for registered_scooter
Best parameters: {'alpha': 0.01, 'max_iter': 3000}
RMSLE score: 0.1675662194215094
```

#### (3) Lasso regression
```
Lasso regression for guest_scooter
Best parameters: {'alpha': 0.01, 'max_iter': 3000}
RMSLE score: 0.27203342609624465

Lasso regression for registered_scooter
Best parameters: {'alpha': 0.01, 'max_iter': 3000}
RMSLE score: 0.1675955071161706
```

Given a test dataset, we can use the Random Forest Regression to predict the future demand of scooters. To get the total demand, we simply add the demand from guest and registered users.

## Moving forward

Time series modelling can be conducted to predict the total demand for scooters.
