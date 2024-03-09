import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from prophet import Prophet
import warnings
import logging
warnings.filterwarnings('ignore')
logger = logging.getLogger('cmdstanpy')
logger.addHandler(logging.NullHandler())
logger.propagate = False
logger.setLevel(logging.CRITICAL)
logging.getLogger('prophet').setLevel(logging.ERROR)


def prophet_model(train_df, test_df):
    pro_train_df = train_df.copy()
    pro_train_df.reset_index(inplace=True)
    pro_train_df.rename(columns={'timestamp':'ds', 'cnt':'y'}, inplace=True)
    pro_train_df['ds'] = pro_train_df['ds'].dt.to_timestamp()

    model = Prophet(weekly_seasonality=True, growth='flat', yearly_seasonality=True, interval_width=0.95, scaling='minmax')
    #Add holiday regressor
    model.add_country_holidays(country_name='UK')
    reg_cols = train_df.drop(['cnt'], axis=1).columns
    for col in reg_cols:
        model.add_regressor(col)

    model.fit(pro_train_df)

    #Predict on test set
    pro_test_df = test_df.copy()
    pro_test_df.reset_index(inplace=True)
    pro_test_df.rename(columns={'timestamp':'ds', 'cnt':'y'}, inplace=True)
    pro_test_df['ds'] = pro_test_df['ds'].dt.to_timestamp()

    y_pred = model.predict(pro_test_df)
    model.plot_components(y_pred)
    plt.show()
    y_pred.set_index('ds', inplace=True)
    return test_df, y_pred

def prophet_metrics(test_df, y_pred):
    rmse = np.sqrt(mean_squared_error(test_df['cnt'], y_pred['yhat']))
    mae = mean_absolute_error(test_df['cnt'], y_pred['yhat'])
    r2 = r2_score(test_df['cnt'], y_pred['yhat'])
    mape = np.mean(np.abs((test_df.to_timestamp()['cnt'] - y_pred['yhat']) / test_df.to_timestamp()['cnt'])) * 100
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def prophet_plot(train_df, test_df, y_pred):
    plt.figure(figsize=(20, 6))
    plt.plot(train_df.index.to_timestamp(), train_df['cnt'], label='Train')
    plt.plot(test_df.index.to_timestamp(), test_df['cnt'], label='Test')
    plt.plot(test_df.index.to_timestamp(), y_pred['yhat'], label='Predictions')
    plt.fill_between(test_df.index.to_timestamp(), y_pred['yhat_lower'], y_pred['yhat_upper'], color='gray', alpha=0.2)
    plt.title('Bike Rentals')
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.title('Actual vs Predicted for Prophet')
    plt.legend()
    plt.show()


