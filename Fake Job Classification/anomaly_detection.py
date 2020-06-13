import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import *
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def estimateGaussian(dataset):
    mu = np.mean(dataset , axis = 0)
    sigma = np.cov(dataset.T)
    return mu, sigma

def multivariateGaussian(dataset , mu , sigma):
    p = multivariate_normal(mean = mu, cov = sigma)
    return p.pdf(dataset)

def selectThresholdByCV(p , y):

    best_epsilon = 0
    best_f1 = 0
    farray = []
    Recallarray = []
    Precisionarray = []
    epsilons = (0.0000e+00, 1.0527717316e-70, 1.0527717316e-50, 1.0527717316e-24)
    for epsilon in epsilons:
        predictions = (p < epsilon)
        f = f1_score(y, predictions, average = "binary")
        Recall = recall_score(y, predictions, average = "binary")
        Precision = precision_score(y, predictions, average = "binary")
        farray.append(f)
        Recallarray.append(Recall)
        Precisionarray.append(Precision)
        print ('For below Epsilon')
        print(epsilon)
        print ('F1 score , Recall and Precision are as below')
        print ('Best F1 Score %f' %f)
        print ('Best Recall Score %f' %Recall)
        print ('Best Precision Score %f' %Precision)
        print ('-'*40)
        if f > best_f1:
            best_f1 = f
            best_recall = Recall
            best_precision = Precision
            best_epsilon = epsilon

    return best_f1, best_epsilon

def choose_feature_importance(X , y):
    clf = RandomForestClassifier(n_estimators = 100 , criterion = 'entropy' , random_state = 0)
    clf.fit(X , y)
    return clf.feature_importances_

def split_data(DataCSV , label):

    train_strip_v1 = DataCSV[DataCSV[label] == 1]
    train_strip_v0 = DataCSV[DataCSV[label] == 0]

    Normal_len = len(train_strip_v0)
    Anomolous_len = len(train_strip_v1)

    start_mid = Anomolous_len // 2
    start_midway = start_mid + 1

    # lấy 50 % anomaly in cross validation
    train_cv_v1 = train_strip_v1[: start_mid]
    train_test_v1 = train_strip_v1[start_midway:Anomolous_len]

    start_mid = (Normal_len * 60) // 100
    start_midway = start_mid + 1

    cv_mid = (Normal_len * 80) // 100
    cv_midway = cv_mid + 1

    # lấy 60 % usual
    train_fraud = train_strip_v0[:start_mid]
    train_cv = train_strip_v0[start_midway:cv_mid]
    train_test = train_strip_v0[cv_midway:Normal_len]

    train_cv = pd.concat([train_cv, train_cv_v1], axis=0)
    train_test = pd.concat([train_test, train_test_v1], axis=0)

    train_cv_y = train_cv[label]
    train_test_y = train_test[label]

    train_cv.drop(labels=[label], axis=1, inplace=True)
    train_fraud.drop(labels=[label], axis=1, inplace=True)
    train_test.drop(labels=[label], axis=1, inplace=True)

    return train_fraud , train_cv , train_cv_y , train_test , train_test_y

def anomaly_detection_analysis(train_fraud , train_cv , train_cv_y , train_test , train_test_y):
    mu, sigma = estimateGaussian(train_fraud)
    p = multivariateGaussian(train_fraud, mu, sigma)
    p_cv = multivariateGaussian(train_cv, mu, sigma)
    p_test = multivariateGaussian(train_test, mu, sigma)
    fscore, ep = selectThresholdByCV(p, train_cv_y)
    predictions = (p_test < ep)
    Recall = recall_score(train_test_y, predictions, average="binary")
    Precision = precision_score(train_test_y, predictions, average="binary")
    F1score = f1_score(train_test_y, predictions, average="binary")
    print('F1 score , Recall and Precision for Test dataset')
    print('Best F1 Score %f' % F1score)
    print('Best Recall Score %f' % Recall)
    print('Best Precision Score %f' % Precision)
    predictions = (p_cv < ep)
    Recall = recall_score(train_cv_y, predictions, average="binary")
    Precision = precision_score(train_cv_y, predictions, average="binary")
    F1score = f1_score(train_cv_y, predictions, average="binary")
    print('F1 score , Recall and Precision for Cross Validation dataset')
    print('Best F1 Score %f' % F1score)
    print('Best Recall Score %f' % Recall)
    print('Best Precision Score %f' % Precision)
