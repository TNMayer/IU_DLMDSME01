from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, recall_score, roc_curve, auc
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pickle

def dropColumns(data, columns = ['tmsp', 'tmsp_hour', 'daytime', 'time', 'failedPSP', 'amountgroup', 'amountgroup_word', 'lower', 'upper', 'numUpper']):
    out = data.copy()
    if len(set(columns).intersection(set(out.columns))) > 0:
        columns = list(set(columns).intersection(set(out.columns)))
        out = out.drop(columns, axis = 1)
    
    return out

def loadPickle(path):
    out = pickle.load(open(path, 'rb'))
    return out

def writePickle(path, object):
    pickle_out = open(path, "wb")
    pickle.dump(object, pickle_out)
    pickle_out.close()
    
def validate_classifier(classifier, X_validate, y_validate, selected_features = []):
    if str(type(classifier)) == "<class 'keras.engine.sequential.Sequential'>":
        prob_predictions = [x[0] for x in list(dnn.predict(X_validate))]
    else:
        prob_predictions = classifier.predict_proba(X_validate[selected_features].values)[:, 1]
    
    if str(type(classifier)) == "<class 'keras.engine.sequential.Sequential'>":
        class_predictions = [x[0] for x in list(np.where(dnn.predict(X_validate) >= 0.5, 1, 0))]
    else:
        class_predictions = classifier.predict(X_validate[selected_features].values)
    
    # print("=== Validation ROC AUC ===")
    # print(roc_auc_score(y_validate, prob_predictions))
    # print("=== Validation Precision ===")
    # print(precision_score(y_validate, class_predictions))
    # print("=== Validation Recall ===")
    # print(recall_score(y_validate, class_predictions))
    # print("=== Validation Accuracy ===")
    # print(accuracy_score(y_validate, class_predictions))
    
    return roc_auc_score(y_validate, prob_predictions)

def keepAllPSP(features):
    psps = ['PSP_Moneycard', 'PSP_Simplecard', 'PSP_UK_Card']
    diff = list(set(psps) - set(features))
    
    print("=== Features to add: " + str(diff) + " ===")
    
    for feature in diff:
        features.append(feature)
    
    return features

def correlationFiltering(X_train, threshold = 0.75, figsize = 10):
    
    plt.figure(figsize=(figsize, figsize))
    sns.heatmap(X_train.corr().round(2), annot=False)
    plt.show()
    
    # create correlation matrix
    corrMatrix = X_train.corr().abs()
    # get upper triangle
    upperCorrMatrix = corrMatrix.where(
        np.triu(np.ones(corrMatrix.shape), k=1).astype(np.bool_))
    uniqueCorrPairs = upperCorrMatrix.unstack().dropna()
    sortedCorrPairs = uniqueCorrPairs.sort_values(ascending = False)
    # identify all paird with correlation greater than threshold
    pairsToFilter = sortedCorrPairs[sortedCorrPairs > threshold]
    toRemove = []
    for pair in pairsToFilter.index:
        # calculate average correlation between A and other variables and B with other variables
        a = pair[0]
        a_avg = corrMatrix[a].mean()
        b = pair[1]
        b_avg = corrMatrix[b].mean()
        # if A has a larger average correlation, remove it, otherwise remove B
        if a_avg > b_avg:
            toRemove.append(a)
        else:
            toRemove.append(b)

    return list(set(toRemove))

def RFECV_ranking(
                    X_train, y_train, X_validate, y_validate, 
                    metrics = ["roc_auc", "precision", "recall"], 
                    step = 1,
                    estimator = RandomForestClassifier(n_estimators=500, random_state=1977),
                    write_model = True
                 ):
    X = pd.concat([X_train, X_validate])
    y = pd.concat([y_train, y_validate]).values.ravel()
    
    train_indices = list(range(len(X_train)))
    test_indices = list(range(len(X_train), len(X)))
    
    cv = [(train_indices, test_indices)]
    importantFeatures = {}
    
    for metric in metrics:
        print("=== " + metric + " ===")
        rfecv = RFECV(estimator=estimator, step=step, cv=cv, scoring=metric, verbose=3)
        rfecv.fit(X,y)
        print('Optimal number of features: {}'.format(rfecv.n_features_))
        try:
            plt.figure(figsize=(16, 9))
            plt.title('Recursive Feature Elimination with Cross-Validation', fontsize=18, fontweight='bold', pad=20)
            plt.xlabel('Number of features selected', fontsize=14, labelpad=20)
            plt.ylabel('Mean ' + metric, fontsize=14, labelpad=20)
            plt.ylim(0, 1)
            plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), rfecv.cv_results_['mean_test_score'], color='#303F9F', linewidth=3)
            plt.show()
        except:
            print("=== There was somewhere an Error ===")

        features = [f for f,s in zip(X_train.columns, rfecv.support_) if s]
        print("= Features:")
        print(features)
        print("= Maximum Test Score:")
        print(rfecv.cv_results_['mean_test_score'].max())
        
        if write_model:
            writePickle(path = "./data/rfecv_" + metric + ".pkl", object = rfecv)
            print("=== rfecv_" + metric + " saved to pickle file ===")
            print("")
        
        importantFeatures[metric] = features
        
    return (rfecv, importantFeatures)

def applyFeatureSelection(
        X_train, y_train, X_validate, y_validate, X_test, y_test,
        metrics = ["roc_auc", "precision", "recall"],
        estimator = RandomForestClassifier(n_estimators=250, random_state=1977),
        correlation_threshold = 0.75,
        step = 1,
        figsize = 10,
        write_model = True,
        correlation_filtering = True
    ):
    
    if X_train is not None:
        if correlation_filtering:
            removeFirst = correlationFiltering(X_train, threshold = correlation_threshold, figsize = figsize)
            print("=== Highly correlated variables to drop ===")
            print(removeFirst)
            X_train = dropColumns(data = X_train, columns = removeFirst)
            X_validate = dropColumns(data = X_validate, columns = removeFirst)
            X_test = dropColumns(data = X_test, columns = removeFirst)
            print("=== Columns after removal ===")
            print(list(X_train.columns))
        model, importantFeatures = RFECV_ranking(X_train, y_train, X_validate, y_validate, 
                                          metrics = metrics, estimator = estimator, 
                                          step = step, write_model = write_model)
        print("=== Most important features ===")
        print(importantFeatures)
    else:
        print("No data to select from")
    
    return model, importantFeatures