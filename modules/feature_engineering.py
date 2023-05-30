import numpy as np
import pandas as pd
import itertools
from sklearn.preprocessing import MinMaxScaler

def getMeanRollingEvent(x):
    if (len(x) > 1):
        result = np.mean(list(x)[:-1])
    else:
        result = np.nan
    
    return result

def getOverallSR(data, train_split = 0.7):
    out = data.copy()
    train_split = int(np.round(len(out)*train_split))
    out["overallSR"] = out.iloc[:train_split, :].success.mean()
                
    return out

def combinatoric_SR(data, addColumns = ["PSP", "card", "3D_secured", "amountgroup_word"], train_split = 0.7):
    out = data.copy()

    train_split = int(np.round(len(out)*train_split))
    combinations = {}
    colName = ""
    for col in addColumns:
        combinations[col] = list(out[col].unique())
        colName = colName + col + "_"

    colName = colName + "SR"
    print(colName)
    addColumns.append(colName)

    keys, values = zip(*combinations.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    joinFrame = pd.DataFrame()

    i = 1
    for permutation in permutations_dicts:
        subset = out.copy().iloc[:train_split, :]
        for key in permutation.keys():
            subset = subset[subset[key] == permutation[key]]
        subset[colName] = subset.success.mean()
        joinFrame = pd.concat([joinFrame, subset[addColumns]])

    out = out.merge(joinFrame.drop_duplicates(), how = 'left', on = list(set(out.columns).intersection(set(joinFrame.columns))))
    
    return out

def combinatoric_event_window_SR(data, 
                                 addColumns = ["PSP", "card", "3D_secured", "amountgroup_word"], 
                                 event_windows = [5, 10, 100, 200],
                                 allowed_missing = 0.05
                                ):
    out = data.copy()

    for event_window in event_windows:
        print("= Event window size: " + str(event_window))
        combinations = {}
        colName = ""
        replaceCol = ""
        for col in addColumns:
            combinations[col] = list(out[col].unique())
            colName = colName + col + "_"
            replaceCol = replaceCol + col + "_"

        colName = colName + "e" + str(event_window) + "_SR"
        replaceCol = replaceCol + "SR"
        print(colName)
        outCols = addColumns.copy()
        outCols.append(colName)

        keys, values = zip(*combinations.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        joinFrame = pd.DataFrame()

        for permutation in permutations_dicts:
            subset = out.copy()
            for key in permutation.keys():
                subset = subset[subset[key] == permutation[key]]
            subset[colName] = subset.success.shift().rolling(
                event_window, min_periods=int(np.ceil(event_window/10))
            ).mean()
            joinFrame = pd.concat([joinFrame, subset[outCols]])
        
        missing_ratio = joinFrame.isna().sum().sum()/len(joinFrame)
        if missing_ratio <= allowed_missing:
            out = out.join(joinFrame[colName])
            out[colName] = out[colName].fillna(out[replaceCol])
        else:
            print("--- Number of missing values too large ---")
    
    return out

def combinatoric_time_window_SR(data,
                                addColumns = ["PSP", "card", "3D_secured", "amountgroup_word"], 
                                time_windows = [1, 6, 12, 24, 72],
                                allowed_missing = 0.15
                                ):
    out = data.copy()

    for time_window in time_windows:
        print("= Time window size: " + str(time_window) + "h")
        combinations = {}
        colName = ""
        replaceCol = ""
        for col in addColumns:
            combinations[col] = list(out[col].unique())
            colName = colName + col + "_"
            replaceCol = replaceCol + col + "_"

        colName = colName + "t" + str(time_window) + "h_SR"
        replaceCol = replaceCol + "SR"
        print(colName)
        outCols = addColumns.copy()
        outCols.append(colName)

        keys, values = zip(*combinations.items())
        permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

        joinFrame = pd.DataFrame()

        for permutation in permutations_dicts:
            subset = out.copy()
            for key in permutation.keys():
                subset = subset[subset[key] == permutation[key]]
            subset[colName] = subset[["tmsp", "success"]].rolling(
                            str(time_window) + "h", on = "tmsp", min_periods=int(np.min([np.round(time_window/10), 3]))
                        ).apply(getMeanRollingEvent)["success"]
            joinFrame = pd.concat([joinFrame, subset[outCols]])

        missing_ratio = joinFrame.isna().sum().sum()/len(joinFrame)
        if missing_ratio <= allowed_missing:
            out = out.join(joinFrame[colName])
            out[colName] = out[colName].fillna(out[replaceCol])
        else:
            print("--- Number of missing values too large: " + str(missing_ratio) + " ---")
    
    return out

def getColumnsToScale(data):
    from pandas.api.types import is_numeric_dtype
    out = []
    for column in data.copy().columns:
        if is_numeric_dtype(data.copy()[column]):
            if data.copy()[column].max() > 1:
                out.append(column)
        else:
            print("Column " + column + " is not numeric")
    
    return out

def dropColumns(data, columns = ['tmsp', 'tmsp_hour', 'daytime', 'time', 'failedPSP', 'amountgroup_word', 'lower', 'upper', 'numUpper']):
    out = data.copy()
    if len(set(columns).intersection(set(out.columns))) > 0:
        columns = list(set(columns).intersection(set(out.columns)))
        out = out.drop(columns, axis = 1)
    
    return out

def applyTimeSplitting(data, train_size = 0.7, test_size = 0.15, validate_size = 0.15, time_col = 'tmsp'):
    applyData = data.copy()
    length = len(applyData)
    
    train_length = int(np.round(length*train_size))
    test_length = int(length - train_length)
    validate_length = int(np.round(test_length * (validate_size/(validate_size + test_size))))
    test_length = int(test_length - validate_length)
    
    y = applyData['success']
    X = dropColumns(applyData, columns = ["success", time_col])
    
    assert (test_length + validate_length + train_length) == length, f"number expected: {length}, got: {test_length + validate_length + train_length}"
    
    X_train = X.copy().iloc[:train_length, :]
    y_train = y.copy().iloc[:train_length]
    X_validate = X.copy().iloc[train_length:(train_length + validate_length), :]
    y_validate = y.copy().iloc[train_length:(train_length + validate_length)]
    X_test = X.copy().iloc[(train_length + validate_length):, :]
    y_test = y.copy().iloc[(train_length + validate_length):]
    
    assert (len(X_train) + len(X_validate) + len(X_test)) == length, f"number expected: {length}, got: {(len(X_train) + len(X_validate) + len(X_test))}"
    
    scale_columns = getColumnsToScale(X)
    scaler = MinMaxScaler()
    X_train[scale_columns] = scaler.fit_transform(X_train[scale_columns])
    X_test[scale_columns] = scaler.transform(X_test[scale_columns])
    if validate_size > 0:
        X_validate[scale_columns] = scaler.transform(X_validate[scale_columns])
    
    print("= Success rate in y_train: " + str(y_train.mean()))
    if validate_size > 0:
        print("= Success rate in y_validate: " + str(y_validate.mean()))
    print("= Success rate in y_test: " + str(y_test.mean()))
    
    return (X, y, X_train, y_train, X_validate, y_validate, X_test, y_test)