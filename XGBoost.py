import pandas as pd
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import optuna
import warnings
warnings.filterwarnings('ignore')


def xgboost_model(train_train_df, val_df, test_df):
    X_train = train_train_df.drop(['cnt'], axis=1)
    y_train = train_train_df['cnt']

    X_val = val_df.drop(['cnt'], axis=1)
    y_val = val_df['cnt']

    X_test = test_df.drop(['cnt'], axis=1)
    y_test = test_df['cnt']

    #Standardize the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    #XGBoost tuning using optuna
    def xgboost_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1100, step=100),
            'max_depth': trial.suggest_int('max_depth', 3, 40),
            'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.1, log=True),
            'subsample': trial.suggest_float('subsample', 0.1, 1.0, step=0.1),
            'random_state': 42
        }
        
        model = XGBRegressor(early_stopping_rounds=10, eval_metric=mean_squared_error, **params)
        model.fit(X_train_scaled, y_train, 
                eval_set=[(X_val_scaled, y_val)],
                verbose=False)
        
        y_pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val, y_pred)
        return mse
        
    study_name = 'xgboost_study'
    #Delete the study if it exists
    try:
        optuna.delete_study(study_name = study_name, storage=f'sqlite:///{study_name}.db')
    except:
        pass
    storage = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(xgboost_objective, n_trials=50, n_jobs=1, show_progress_bar=True)

    # Best parameters
    best_params = study.best_params
    print('Best parameters:', best_params)

    #Train the model with best parameters
    model = XGBRegressor(verbosity=1, **best_params)
    model.fit(X_train_scaled, y_train)

    #Predict on test set
    y_pred = model.predict(X_test_scaled)
    return y_train, y_test, y_pred

def xgboost_metrics(y_test, y_pred):
    #RMSE, MAE, MAPE, R2
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def xgboost_plot(train_df, test_df, y_pred):
    plt.figure(figsize=(20, 6))
    plt.plot(train_df.index.to_timestamp(), train_df['cnt'], label='Train')
    plt.plot(test_df.index.to_timestamp(), test_df['cnt'], label='Test')
    plt.plot(test_df.index.to_timestamp(), y_pred, label='Predictions')
    plt.title('Bike Rentals')
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.legend()
    plt.show()
