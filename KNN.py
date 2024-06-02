import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    train_data = pd.read_csv('./data/training.csv')
    test_data = pd.read_csv('./data//test_X.csv')

    X_train = train_data.drop(columns=['lettr'])
    X_test = test_data

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=6)
    knn.fit(X_train_scaled, np.ones(X_train_scaled.shape[0]))

    distances, _ = knn.kneighbors(X_test_scaled)

    outliers = distances.max(axis=1)

    submission = pd.DataFrame({
        'id': np.arange(len(outliers)),
        'outliers': outliers
    })

    submission.to_csv('./submission/knn_submission.csv', index=False)