from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense

def build_gru_model(X_train):
    model = Sequential([
        GRU(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        GRU(32),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    return model
