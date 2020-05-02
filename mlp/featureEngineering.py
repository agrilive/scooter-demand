import pandas as pd
import numpy as np
import datetime  
from datetime import date 
import calendar 
import os
import graphviz
from sklearn import tree

def cleanWeather(word):
    """Corrects the spelling error for the weather variable"""
    word = word.lower()
    if word == 'lear' or word == 'clar':
        return 'clear'
    elif word == 'loudy' or word == 'cludy':
        return 'cloudy'
    elif word == 'liht snow/rain':
        return 'light snow/rain'
    else:
        return word

def findDay(date): 
    """Returns the day of the week based on date"""
    year, month, day = (int(i) for i in date.split('-'))     
    dt = datetime.date(year, month, day) 
    return dt.strftime("%A")

def binDay(day):
    """Groups days of the week into 2 bins - weekends and weekdays"""
    if day == 'Saturday' or day == 'Sunday':
        return 1
    else:
        return 0

def binYear(df):
    """Groups the year + its quarter into bins"""
    if df.year == 2011 and df.month <= 3:
        return 1
    elif df.year == 2011 and df.month <= 6:
        return 2
    elif df.year ==2011 and df.month <= 9:
        return 3
    elif df.year == 2011 and df.month <= 12:
        return 4
    elif df.year == 2012 and df.month <= 3:
        return 5
    elif df.year == 2012 and df.month <= 6:
        return 6
    elif df.year == 2012 and df.month <= 9:
        return 7
    else:
        return 8

def numWeather(word):
    """Converts weather to numerical value"""
    if word == 'clear':
        return 1
    elif word == 'cloudy':
        return 2
    elif word == 'light snow/rain':
        return 3
    else:
        return 4

def decisionTreeBinning(X, y, lab1, lab2):
    """Uses decision tree to identify bins for hr and temperature variable
    ----------
    Returns: PDF file with image of decision tree"""
    clf = tree.DecisionTreeRegressor(max_depth=3, min_samples_split=0.3, random_state=42)
    clf.fit(X, y)
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = graphviz.Source(dot_data) 
    return graph.render("output/bin{0}_{1}".format(lab1, lab2)) 

def binHrGue(hr):
    """Splits hour variable into bins for guest users"""
    if hr <= 6:
        return 1
    elif hr == 7:
        return 2
    elif hr == 8:
        return 3
    elif hr <= 10:
        return 4
    elif hr <= 19:
        return 5
    else:
        return 6

def binHrReg(hr):
    """Splits hour variable into bins for registered users"""
    if hr <= 6: 
        return 1
    elif hr <= 16:
        return 2
    elif hr <= 20:
        return 3
    else:
        return 4

def binTempGue(temp):
    """Splits temperature variable into bins for guest users"""
    if temp <= 76.05:
        return 1
    elif temp <= 86.15:
        return 2
    elif temp <= 99.7:
        return 3
    elif temp <= 106.45:
        return 4
    else:
        return 5

def binTempReg(temp):
    """Splits temperature variable into bins for registered users"""
    if temp <= 70.95:
        return 1
    elif temp <= 86.15:
        return 2
    elif temp <= 104.75:
        return 3
    elif temp <= 106.45:
        return 4
    else:
        return 5

class FeatureEngineering:
    """
    Data pre-processing steps (e.g. clean for spelling error) and create new variables/ reduce feature levels 
    Parameters
    ----------
    dataset_folder: directory of csv file
    filename: csv file
    """

    def __init__(self, dataset_folder, filename):
        self.dataset_folder = dataset_folder
        self.filename = filename

    def load_file(self):
        self.df = pd.read_csv(os.path.join(self.dataset_folder, self.filename))
        self.df.columns = self.df.columns.str.strip()
        self.df = self.df.drop_duplicates()

        # index by date and time
        self.df['hr'] = [format(i, "02") for i in self.df['hr']] 
        self.df['date_hr'] = pd.to_datetime(self.df['date'] + ' ' + self.df['hr'], format = '%Y%m%d %H')
        self.df = self.df.set_index('date_hr').sort_index()
        self.df['hr'] = self.df['hr'].astype(int)
        return self.df

    def clean_values(self):
        # create year and month variables
        self.df['year'] = self.df.index.year
        self.df['month'] = self.df.index.month

        # forward fill for no. of users < 0 
        self.df[self.df[['guest_scooter','registered_scooter']] < 0] = np.nan
        self.df[['guest_scooter','registered_scooter']] = self.df[['guest_scooter','registered_scooter']].ffill(axis = 0) 

        # log transform the number of users 
        self.df['guest_log'] = np.log1p(self.df['guest_scooter'])
        self.df['registered_log'] = np.log1p(self.df['registered_scooter']) 

        # apply functions for weather variable
        self.df['weather'] = self.df['weather'].apply(cleanWeather)
        self.df['weather'] = self.df['weather'].apply(numWeather)
        return self.df

    def find_bins(self):
        # make bins for hr and temperature variable
        if not os.path.exists('output'):
            os.makedirs('output')
    
        X_hr = self.df['hr'].values.reshape(-1,1)
        X_temp = self.df['temperature'].values.reshape(-1,1)
        y_gue = self.df['guest_scooter'].values.reshape(-1,1)
        y_reg = self.df['registered_scooter'].values.reshape(-1,1)

        decisionTreeBinning(X_hr, y_gue, 'hr', 'guest')
        decisionTreeBinning(X_hr, y_reg, 'hr', 'registered')
        decisionTreeBinning(X_temp, y_gue, 'temp', 'guest')
        decisionTreeBinning(X_temp, y_reg, 'temp', 'registered')
        print('Check output folder for decision trees and define functions to bin variables')

    def new_variables(self):
        self.df['day'] = [findDay(i) for i in self.df['date']]    
        self.df['day_bin'] = self.df['day'].apply(binDay)
        self.df['year_bin'] = self.df.apply(binYear, axis=1)

        self.df['hr_gue'] = self.df['hr'].apply(binHrGue)
        self.df['hr_reg'] = self.df['hr'].apply(binHrReg)
        self.df['temp_gue'] = self.df['temperature'].apply(binTempGue)
        self.df['temp_reg'] = self.df['temperature'].apply(binTempReg)
        return self.df

    def save_file(self, new_fn):
        # save DataFrame to a new csv file
        return self.df.to_csv(os.path.join(self.dataset_folder, new_fn))
