import pytest
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_test_data():
    # テスト用のダミーデータを作成
    data = np.random.rand(100, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

def create_lstm_model(time_step):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(time_step, 1)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def create_dataset(dataset, time_step):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

def test_model_training():
    data, scaler = create_test_data()
    time_step = 60
    X, Y = create_dataset(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = create_lstm_model(time_step)
    model.fit(X, Y, epochs=1, batch_size=1, verbose=0)
    
    assert model is not None

def test_model_prediction():
    data, scaler = create_test_data()
    time_step = 60
    X, Y = create_dataset(data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    model = create_lstm_model(time_step)
    model.fit(X, Y, epochs=1, batch_size=1, verbose=0)
    
    predictions = model.predict(X)
    assert predictions.shape == (len(X), 1)