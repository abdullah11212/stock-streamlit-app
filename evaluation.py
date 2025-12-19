import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate(model, X, y, scaler, scaled_features):
    preds = model.predict(X, verbose=0)
    temp = np.zeros((len(preds), scaled_features.shape[1]))
    temp[:,0] = preds[:,0]
    preds_inv = scaler.inverse_transform(temp)[:,0]

    actual = np.zeros_like(temp)
    actual[:,0] = y
    actual_inv = scaler.inverse_transform(actual)[:,0]

    return {
        "MAE": mean_absolute_error(actual_inv, preds_inv),
        "RMSE": mean_squared_error(actual_inv, preds_inv, squared=False),
        "R2": r2_score(actual_inv, preds_inv)
    }
