import pandas as pd
import numpy as np
import time
import os
import pickle
from jenkspy import JenksNaturalBreaks
import sqlite3
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc

def swap_rows(df, row1, row2):
    df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
    return df

def find_diff(df_a, df_b, column):
    listA = list(df_a[column])
    listB = list(df_b[column])
    # index variable
    idx = 0

    # Result list
    res = []

    # With iteration
    for i in listA:
        if i != listB[idx]:
            res.append(idx)
        idx = idx + 1
    
    return res

def convert_types(X, y):
    out_X = X.copy()
    out_y = y.copy()
    for col in out_X.columns:
        if (len(out_X[col].unique()) == 2):
            out_X[col] = out_X[col].astype(int)
    for col_y in out_y.columns:
        out_y[col_y] = out_y[col_y].astype(int)
    
    return (out_X, out_y)

def get_amountgroups(data, bins = [0, 99, 175, 247, 330, 8000]):
    # [6, 99, 175, 247, 330, 630]
    labels = ['very low', 'low', 'medium', 'high', 'very high']
    out = data.copy()
    out['amountgroup_word'] = pd.cut(x = out['amount'], bins = bins, labels = labels, include_lowest = False)
    
    return out

def get_daytime(data):
    out = data.copy()
    
    out['daytime'] = np.where((out['time'] >= '01:00') & (out['time'] <= '06:00'), 'night', 
                        np.where((out['time'] > '06:00') & (out['time'] <= '12:00'), 'morning',
                        np.where((out['time'] > '12:00') & (out['time'] <= '18:00'), 'afternoon', 'evening')))
    
    return out

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

def writeDb(data, pathDb, table_name):
    conn = sqlite3.connect(pathDb)
    try:
        data.copy().to_sql(table_name, conn, index_label = "index", if_exists = "fail")
        print("=== Table " + table_name + " created successful ===")
    except:
        print("=== Table already exists and will not be replaced ===")
    conn.close()

def checkIfTableDbExists(pathDb, table_name):
    conn = sqlite3.connect(pathDb)
    try:
        test = pd.read_sql("SELECT * FROM " + table_name + " LIMIT 1", conn)
        out = True
    except:
        print("=== Table does not exists ===")
        out = False
    conn.close()
    
    return out

def readSqlTable(pathDb, table):
    conn = sqlite3.connect(pathDb)
    out = pd.read_sql(f"SELECT * FROM {table}", conn)
    out.index = out["index"]
    out.index.name = None
    if 'tmsp' in out.columns:
        out["tmsp"] = pd.to_datetime(out["tmsp"])
    out = dropColumns(out, columns = ['index'])
    conn.close()
    
    return out

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
    
def saveSplittings(X, y, X_train, y_train, X_validate, y_validate, X_test, y_test,
                  pathList = ['./data/X_train.csv', './data/X_test.csv', './data/y_train.csv',
                             './data/y_test.csv', './data/X_validate.csv', './data/y_validate.csv',
                             './data/X.csv', './data/y.csv']
                  ):
    try:
        X_train.to_csv(pathList[0], index = True,  index_label = 'index')
        X_test.to_csv(pathList[1], index = True, index_label = 'index')
        y_train.to_csv(pathList[2], index = True, index_label = 'index')
        y_test.to_csv(pathList[3], index = True, index_label = 'index')
        X_validate.to_csv(pathList[4], index = True, index_label='index')
        y_validate.to_csv(pathList[5], index = True, index_label='index')
        X.to_csv(pathList[6], index = True, index_label = 'index')
        y.to_csv(pathList[7], index = True, index_label = 'index')
        print("=== X, y, X_train, y_train, X_validate, y_validate, X_test, y_test successfully written to disk ===")
    except:
        print("=== There was a serious error ===")

def getMeanRollingEvent(x):
    if (len(x) > 1):
        result = np.mean(list(x)[:-1])
    else:
        result = np.nan
    
    return result

def getMeanRollingTime(x):
    result = np.mean(x[:-1])
    return result

def applyFunc(x, func = np.mean):
    if (len(x) > 1):
        result = func(list(x)[:-1])
    else:
        result = np.nan
    
    return result

def dropColumns(data, columns = ['tmsp', 'tmsp_hour', 'daytime', 'time', 'failedPSP', 'amountgroup_word', 'lower', 'upper', 'numUpper']):
    out = data.copy()
    if len(set(columns).intersection(set(out.columns))) > 0:
        columns = list(set(columns).intersection(set(out.columns)))
        out = out.drop(columns, axis = 1)
    
    return out

def find_threshold(X_test, y_test, model, features, range_min = 30, range_max = 70, precision_weight = 2, recall_weight = 1):
    thresholds = np.array(range(range_min, (range_max + 1)))/100
    optimized = 0
    precision_max = 0
    recall_max = 0
    optim_threshold = 0
    for threshold in thresholds:
        performance = validate_classifier(model, X_test, y_test, selected_features = features, threshold = threshold, verbose = 0)
        precision = performance['precision']
        recall = performance['recall']
        if ((precision_weight*precision + recall_weight*recall)/(precision_weight + recall_weight)) > optimized:
            precision_max = precision
            recall_max = recall
            optim_threshold = threshold
            optimized = (precision_weight*precision + recall_weight*recall)/(precision_weight + recall_weight)
    
    return (optim_threshold, precision_max, recall_max)

def validate_classifier(classifier, X_validate, y_validate, selected_features = [], threshold = 0.5, verbose = 1):
    if str(type(classifier)) == "<class 'keras.engine.sequential.Sequential'>":
        prob_predictions = [x[0] for x in list(dnn.predict(X_validate))]
    else:
        prob_predictions = classifier.predict_proba(X_validate[selected_features])[:, 1]
    
    if str(type(classifier)) == "<class 'keras.engine.sequential.Sequential'>":
        class_predictions = [x[0] for x in list(np.where(dnn.predict(X_validate) >= threshold, 1, 0))]
    else:
        class_predictions = list(np.where(classifier.predict_proba(X_validate[selected_features])[:, 1] >= threshold, 1, 0))
    
    if verbose > 0:
        print("=== Validation ROC AUC ===")
        print(roc_auc_score(y_validate, prob_predictions))
        print("=== Validation Precision ===")
        print(precision_score(y_validate, class_predictions, zero_division = 0))
        print("=== Validation Recall ===")
        print(recall_score(y_validate, class_predictions))
        print("=== Validation Accuracy ===")
        print(accuracy_score(y_validate, class_predictions))
    
    return {
                "roc_auc": roc_auc_score(y_validate, prob_predictions),
                "precision": precision_score(y_validate, class_predictions, zero_division = 0),
                "recall": recall_score(y_validate, class_predictions),
                "accuracy": accuracy_score(y_validate, class_predictions)
           }

def formatData(data, columns = ['PSP', 'card', 'month', 'weekday', 'country']):
    out = data.copy()
    
    out = pd.get_dummies(out, columns=columns, drop_first=True)
    # print("=== Number of missing values ===")
    # print(out.isna().sum().sum())
    
    return out