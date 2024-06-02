import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

if __name__ == '__main__':
    train_data = pd.read_csv('./data/training.csv')
    test_data = pd.read_csv('./data//test_X.csv')

    X_train = train_data.drop(columns=['lettr'])
    X_test = test_data

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm = OneClassSVM(kernel='poly', degree=6, gamma='scale', coef0=1, nu=0.1)
    svm.fit(X_train_scaled)

    outliers = svm.decision_function(X_test_scaled)

    submission = pd.DataFrame({
        'id': np.arange(len(outliers)),
        'outliers': outliers
    })

    submission.to_csv('./submission/svm_submission.csv', index=False)