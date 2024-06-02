import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam


class Autoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.0008):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.model = self.build_model()
        
    def build_model(self):
        input_layer = Input(shape=(self.input_dim,))
        encoder = Dense(self.encoding_dim, activation="relu")(input_layer)
        encoder = Dense(int(self.encoding_dim / 2), activation="relu")(encoder)
        encoder = Dense(int(self.encoding_dim / 4), activation="relu")(encoder)
        decoder = Dense(int(self.encoding_dim / 2), activation="relu")(encoder)
        decoder = Dense(self.encoding_dim, activation="relu")(decoder)
        decoder = Dense(self.input_dim, activation="linear")(decoder)
        
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        optimizer = Adam(learning_rate=self.learning_rate)
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        return autoencoder
    
    def train(self, X_train, epochs=1000, batch_size=64, validation_split=0.1):
        self.model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=validation_split, verbose=1)
    
    def predict(self, X):
        return self.model.predict(X)


if __name__ == '__main__':
    train_data = pd.read_csv('./data/training.csv')
    test_data = pd.read_csv('./data//test_X.csv')

    X_train = train_data.drop(columns=['lettr'])
    X_test = test_data

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    input_dim = X_train_scaled.shape[1]
    encoding_dim = 42 

    model = Autoencoder(input_dim=input_dim, encoding_dim=encoding_dim)
    model.train(X_train_scaled, epochs=1000, batch_size=64, validation_split=0.1)

    X_test_pred = model.predict(X_test_scaled)
        
    outliers = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

    submission = pd.DataFrame({
        'id': np.arange(len(outliers)),
        'outliers': outliers
    })

    submission.to_csv('./submission/autoencoder_submission.csv', index=False)