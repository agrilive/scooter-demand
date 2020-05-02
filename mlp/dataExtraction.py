import pyodbc
import csv
import os

class QueryDatabase:

    def __init__(self, columns):
        self.columns = columns

    def connect(self, connstring, querystring):
        conn = pyodbc.connect(connstring)
        self.cursor = conn.cursor()
        self.cursor.execute(querystring)
        self.rows = self.cursor.fetchall()
        return self.rows

    def save_csv(self, folder, filename):
        ind_col = self.columns.split(',')
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.path = os.path.join(folder, filename)

        with open(self.path, 'w', newline= '') as f:
            a = csv.writer(f, delimiter=',')
            a.writerow(ind_col)  
            a.writerows(self.rows) 

        self.cursor.close()
        print('Extracted data and saved into csv file')
