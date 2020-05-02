import os
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

from dataExtraction import QueryDatabase
from featureEngineering import FeatureEngineering
from trainTestSplit import TimeSSplit
from regression import RegressionModel

# needed_cols = 'date, hr, weather, temperature, feels_like_temperature, relative_humidity, windspeed, psi, guest_scooter, registered_scooter'
# connstring - removed as per requested by database owner
# querystring = "SELECT date, hr, weather, temperature, feels_like_temperature, relative_humidity, windspeed, psi, guest_scooter, registered_scooter FROM rental_data WHERE (date >= '2011-01-01' and date < '2013-01-01')"
filename = 'rental_data.csv'
new_fn = 'clean_data.csv'
dataset_folder = 'dataset'

# query = QueryDatabase(needed_cols)
# query.connect(connstring, querystring)
# query.save_csv(dataset_folder, filename)

feng = FeatureEngineering(dataset_folder, filename)
feng.load_file()
feng.clean_values()
feng.find_bins()
feng.new_variables()
feng.save_file(new_fn)

df = pd.read_csv(os.path.join(dataset_folder, new_fn), index_col='date_hr')
gue_features = ['year_bin','day_bin','hr_gue','weather','temp_gue','relative_humidity','windspeed','psi']
reg_features = ['year_bin','day_bin','hr_reg','weather','temp_reg','relative_humidity','windspeed','psi']

tss = TimeSSplit()
guest_X = df[gue_features].values
guest_y = df['guest_log'].values
reg_X = df[reg_features].values
reg_y = df['registered_log'].values
gX_train, gX_test, gy_train, gy_test = tss.split(guest_X, guest_y)
rX_train, rX_test, ry_train, ry_test = tss.split(reg_X, reg_y)

lasso_ridge_params = {'max_iter':[3000], 'alpha':[0.01, 0.05, 0.1, 1, 10, 100, 200, 300, 400, 800]}
rf_params = {'n_estimators':[100,200,300,400,500],
                    'max_depth':[4,5,6,7,8,9,10]}

# Lasso regression for guest users
print('\nLasso regression for {0} users'.format('guest'))
glm = RegressionModel(Lasso(), lasso_ridge_params)
glm.grid_fit(gX_train, gy_train)
glm.predict(gX_test, gy_test)

# Ridge regression for guest users
print('\nRidge regression for {0} users'.format('guest'))
grm = RegressionModel(Ridge(), lasso_ridge_params)
grm.grid_fit(gX_train, gy_train)
grm.predict(gX_test, gy_test)

# Random forest regression for guest users
print('\nRandom forest regression for {0} users'.format('guest'))
grf = RegressionModel(RandomForestRegressor(), rf_params)
grf.grid_fit(gX_train, gy_train)
grf.predict(gX_test, gy_test)

# Lasso regression for registered users
print('\nLasso regression for {0} users'.format('registered'))
rlm = RegressionModel(Lasso(), lasso_ridge_params)
rlm.grid_fit(rX_train, ry_train)
rlm.predict(rX_test, ry_test)

# Ridge regression for registered users
print('\nRidge regression for {0} users'.format('registered'))
rrm = RegressionModel(Ridge(), lasso_ridge_params)
rrm.grid_fit(rX_train, ry_train)
rrm.predict(rX_test, ry_test)

# Random forest regression for registered users
print('\nRandom forest regression for {0} users'.format('registered'))
rrf = RegressionModel(RandomForestRegressor(), rf_params)
rrf.grid_fit(rX_train, ry_train)
rrf.predict(rX_test, ry_test)

# Predict the total users based on model which gives lowest RMSLE for guest and registered users
guest_pred = grf.best_model(guest_X, guest_y)
reg_pred = rrf.best_model(reg_X, reg_y)
pred_df = pd.DataFrame(df.index)
pred_df['guest_pred'] = guest_pred
pred_df['registered_pred'] = reg_pred
pred_df['total_pred'] = guest_pred + reg_pred
print(pred_df.head())


