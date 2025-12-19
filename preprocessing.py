import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def prepare_sequences(data, seq_len=60):
    features = data[['Price', 'Volatility']].values

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(seq_len, len(scaled)):
        X.append(scaled[i-seq_len:i])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)

    return train_test_split(X, y, test_size=0.2, shuffle=False) + (scaler, scaled)
