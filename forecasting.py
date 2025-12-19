import numpy as np
import pandas as pd

def future_forecast(model, data, scaler, scaled_features, seq_len, n_days, heston_params):
    last_seq = scaled_features[-seq_len:]
    preds = []

    for _ in range(n_days):
        x = last_seq.reshape(1, seq_len, scaled_features.shape[1])
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        last_seq = np.vstack([last_seq[1:], [p, last_seq[-1,1]]])

    temp = np.zeros((n_days, scaled_features.shape[1]))
    temp[:,0] = preds
    prices = scaler.inverse_transform(temp)[:,0]

    dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=n_days)
    return dates, prices
