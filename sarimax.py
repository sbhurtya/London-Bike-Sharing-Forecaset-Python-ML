import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from pmdarima.arima import StepwiseContext
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def SARIMAX_model(train_df, test_df):
    #Standardize the data
    scaler = MinMaxScaler().set_output(transform="pandas")
    X_train = train_df.drop('cnt', axis=1)
    y_train = train_df['cnt']

    X_test = test_df.drop('cnt', axis=1)
    y_test = test_df['cnt']

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Auto ARIMA
    with StepwiseContext(max_steps=3):
        model = auto_arima(y=y_train,
                            X=X_train_scaled,
                            start_p=0,
                            d=None,
                            start_q=0,
                            max_p=3,
                            max_d=7,
                            max_q=3,
                            start_P=0,
                            D=None,
                            start_Q=0,
                            m=52,
                            max_P=5,
                            max_D=7,
                            max_Q=5,
                            stationary=False,
                            seasonal=True,
                            stepwise = True,
                            random=False,
                            random_state=42,
                            njobs=1,
                            scoring='mse',
                            maxiter=50,
                            trace=True,
                            )
        print(model.summary())

    #Predictions
    y_pred = model.predict(n_periods=len(y_test), X=X_test_scaled)
    y_pred = pd.Series(y_pred, index=y_test.index)
    return y_train, y_test, y_pred,

def SARIMAX_metrics(y_test, y_pred):
    #RMSE, MAE, MAPE, R2
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def SARIMAX_plot(y_train, y_test, y_pred):
    #Plot actual vs predicted
    plt.figure(figsize=(20, 6))
    plt.plot(y_train.index.to_timestamp(), y_train, label='Train')
    plt.plot(y_test.index.to_timestamp(), y_test, label='Actual')
    plt.plot(y_test.index.to_timestamp(), y_pred, label='Predicted')
    plt.xticks(rotation=90)
    plt.legend()
    

