import pandas as pd
import numpy as np
import time
import os
import pickle

def convert_types(X, y):
    out_X = X.copy()
    out_y = y.copy()
    for col in out_X.columns:
        if (len(out_X[col].unique()) == 2):
            out_X[col] = out_X[col].astype(int)
    for col_y in out_y.columns:
        out_y[col_y] = out_y[col_y].astype(int)
    
    return (out_X, out_y)

def loadPickle(path):
    out = pickle.load(open(path, 'rb'))
    return out

def writePickle(path, object):
    pickle_out = open(path, "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()

def get_y(data, groups=["card"]):
    y1 = data.groupby(groups).aggregate({'tmsp': 'count'}).rename(columns={'tmsp': 'count'}).reset_index()
    y2 = data[data["success"] == 1].groupby(groups).aggregate({'tmsp': 'count'}).rename(columns={"tmsp": "count_success"}).reset_index()
    
    y = y1.merge(y2, on = groups)
    y["success_rate"] = y['count_success']/y['count']
    
    return y

def generateDataFromFile():
    if (
        (os.path.isfile('./data/X_train.csv')) & (os.path.isfile('./data/X_test.csv')) & 
        (os.path.isfile('./data/y_train.csv')) & (os.path.isfile('./data/y_test.csv')) &
        (os.path.isfile('./data/y_validate.csv')) & (os.path.isfile('./data/X_validate.csv'))
    ):
        print("generate from files")
        X = pd.read_csv('./data/X.csv', index_col = 'index')
        X.index.name = None
        y = pd.read_csv('./data/y.csv', index_col = 'index')
        y.index.name = None
        X_train = pd.read_csv('./data/X_train.csv', index_col = 'index')
        X_train.index.name = None
        X_test = pd.read_csv('./data/X_test.csv', index_col = 'index')
        X_test.index.name = None
        X_validate = pd.read_csv('./data/X_validate.csv', index_col = 'index')
        X_validate.index.name = None
        y_train = pd.read_csv('./data/y_train.csv', index_col = 'index')
        y_train.index.name = None
        y_test = pd.read_csv('./data/y_test.csv', index_col = 'index')
        y_test.index.name = None
        y_validate = pd.read_csv('./data/y_validate.csv', index_col = 'index')
        y_validate.index.name = None
        
        X, y = convert_types(X, y)
        X_train, y_train = convert_types(X_train, y_train)
        X_validate, y_validate = convert_types(X_validate, y_validate)
        X_test, y_test = convert_types(X_test, y_test)
        
        print("=== all data loaded from files ===")
        
        return (X, y, X_train, y_train, X_validate, y_validate, X_test, y_test)
    else:
        print("skip the step because not all data is prepared yet")
        
        return (None, None, None, None, None, None, None, None)