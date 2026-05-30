from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(x_train, y_train, estimator):
    y_pred = estimator.predict(x_train)
    mse = mean_squared_error(y_train, y_pred)
    mae = mean_absolute_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    return mse, mae, r2
