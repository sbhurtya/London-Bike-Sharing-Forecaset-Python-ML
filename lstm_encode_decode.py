import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_CUDNN_USE_AUTOTUNE'] ='0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


import pandas as pd
import numpy as np
import random
np.random.seed(42)
random.seed(42)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
tf.autograph.set_verbosity(0, alsologtostdout=False)
tf.keras.config.disable_interactive_logging()
from tensorflow.python.keras import backend as K
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import warnings
warnings.filterwarnings('ignore')

def seed_everything():
    tf.get_logger().setLevel('ERROR')
    tf.random.set_seed(42)
    tf.keras.utils.set_random_seed(42)
    tf.compat.v1.reset_default_graph()
    tf.config.experimental.enable_op_determinism()
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    K.set_session(sess)
    K.clear_session()


def lstm_model(train_train_df, val_df, test_df, lookback, forecast_horizon):
    feature_columns = train_train_df.drop('cnt', axis=1).columns
    target_column = 'cnt'

    #Scale the data
    scaler = MinMaxScaler()
    train_df_scaled = train_train_df.copy()
    val_df_scaled = val_df.copy()
    test_df_scaled = test_df.copy()
    train_df_scaled[feature_columns] = scaler.fit_transform(train_df_scaled[feature_columns])
    val_df_scaled[feature_columns] = scaler.transform(val_df_scaled[feature_columns])
    test_df_scaled[feature_columns] = scaler.transform(test_df_scaled[feature_columns])

    def create_dataset(df, n_deterministic_features,
                    window_size, forecast_size,
                    batch_size):
        total_size = window_size + forecast_size

        data = tf.data.Dataset.from_tensor_slices(df.values)
        data = data.window(total_size, shift=1, drop_remainder=True)
        data = data.flat_map(lambda k: k.batch(total_size))
        # data = data.shuffle(shuffle_buffer_size, seed=42)
        data = data.map(lambda k: ((k[:-forecast_size],
                                    k[-forecast_size:, -n_deterministic_features:]),
                                k[-forecast_size:, 0]))

        return data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    lookback = lookback #days
    forecast_horizon = forecast_horizon #day

    number_of_features = len(train_train_df.columns)
    number_of_aleatoric_features = 1 #Only cnt is aleatoric
    number_of_deterministic_features = number_of_features - number_of_aleatoric_features

    batch_size = 1
    training_window = create_dataset(train_df_scaled,
                                    number_of_deterministic_features,
                                    lookback,
                                    forecast_horizon,
                                    batch_size)

    validation_window = create_dataset(val_df_scaled,
                                    number_of_deterministic_features,
                                    lookback,
                                    forecast_horizon,
                                    batch_size)

    testing_window = create_dataset(test_df_scaled,
                                    number_of_deterministic_features,
                                    lookback,
                                    forecast_horizon,
                                    batch_size) 
    
    #Tune using optuna
    def lstm_objective(trial):
        seed_everything()
        latent_dim = trial.suggest_categorical('latent_dim', [16, 32])
        num_layers = trial.suggest_categorical('num_layers', [1, 2, 3])
        dense_units = trial.suggest_categorical('dense_units', [32, 64, 128, 256])
        dropout = trial.suggest_float('dropout', 0.1, 0.5, step=0.1)
        learning_rate = trial.suggest_float('learning_rate', 5e-4, 1e-2, log=True)

        past_inputs = tf.keras.Input(
            shape=(lookback, number_of_features), name='past_inputs')
        # Encoding the past
        encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(past_inputs)

        future_inputs = tf.keras.Input(
            shape=(forecast_horizon, number_of_deterministic_features), name='future_inputs')
        
        decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
        x = decoder_lstm(future_inputs,
                        initial_state=[state_h, state_c])
        x = tf.keras.layers.Dropout(dropout)(x)
        for _ in range(num_layers - 1):
            x = tf.keras.layers.LSTM(latent_dim, return_sequences=True)(x)
            x = tf.keras.layers.Dropout(dropout)(x)

        x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
        x = tf.keras.layers.Dense(int(dense_units/2), activation='relu')(x)

        output = tf.keras.layers.Dense(1, activation='relu')(x)

        model = tf.keras.models.Model(
            inputs=[past_inputs, future_inputs], outputs=output)
        
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                        loss='mse',
                        metrics=['mape'])
        
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            mode='min', 
                                            verbose=0, 
                                            patience=10, 
                                            restore_best_weights=True)
        
        history = model.fit(training_window, 
                            epochs=100, 
                            validation_data=validation_window, 
                            callbacks=[es],
                            verbose=0,
                            shuffle=False
                        )
        return np.min(history.history['val_loss'])

    study_name = 'lstm_study'
    #Delete the study if it exists
    try:
        optuna.delete_study(study_name = study_name, storage=f'sqlite:///{study_name}.db')
    except:
        pass
    storage = f'sqlite:///{study_name}.db'
    study = optuna.create_study(study_name=study_name, storage=storage, load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lstm_objective, n_trials=5, n_jobs=1, show_progress_bar=False)

    # Best parameters
    study_name = 'lstm_study'
    #Load the study
    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{study_name}.db')
    best_params = study.best_params
    seed_everything()


    #Train the model with best parameters
    latent_dim = best_params['latent_dim']
    num_layers = best_params['num_layers']
    dense_units = best_params['dense_units']
    dropout = best_params['dropout']
    learning_rate = best_params['learning_rate']

    past_inputs = tf.keras.Input(
        shape=(lookback, number_of_features), name='past_inputs')
    # Encoding the past
    encoder = tf.keras.layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(past_inputs)

    future_inputs = tf.keras.Input(
        shape=(forecast_horizon, number_of_deterministic_features), name='future_inputs')

    decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True)
    x = decoder_lstm(future_inputs,
                    initial_state=[state_h, state_c])
    x = tf.keras.layers.Dropout(dropout)(x)
    for _ in range(num_layers - 1):
        x = tf.keras.layers.LSTM(latent_dim, return_sequences=True)(x)
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Dense(dense_units, activation='relu')(x)
    x = tf.keras.layers.Dense(int(dense_units/2), activation='relu')(x)

    output = tf.keras.layers.Dense(1, activation='relu')(x)

    model = tf.keras.models.Model(
        inputs=[past_inputs, future_inputs], outputs=output)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                    loss='mse',
                    metrics=['mape'])

    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            mode='min', 
                                            verbose=1, 
                                            patience=20, 
                                            restore_best_weights=True)


    K.clear_session()

    history = model.fit(training_window, 
                        epochs=300, 
                        validation_data=validation_window, 
                        callbacks=[es],
                        verbose=1,
                        shuffle=False
                    )
    
    #Predict on test set
    predictions = []
    actuals = []
    for i, data in enumerate(testing_window):
        if i % 7 == 0:
            (past, future), truth = data
            y_pred = model.predict([past, future], verbose=0)
            y_pred = y_pred.flatten()
            truth = truth.numpy().flatten()
            predictions.append(y_pred)
            actuals.append(truth)

    predictions = [item for sublist in predictions for item in sublist]
    actuals = [item for sublist in actuals for item in sublist]
    y_pred = np.array(predictions)
    y_test = np.array(actuals)
    return y_test, y_pred

def lstm_metrics(actuals, predictions):
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    r2 = r2_score(actuals, predictions)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def lstm_plot(train_df, test_df, predictions, lookback):
    lstm_pred_df = test_df[lookback:lookback+154].copy()
    lstm_pred_df['predictions'] = predictions

    plt.figure(figsize=(20, 6))
    plt.plot(train_df.index.to_timestamp(), train_df['cnt'], label='Train')
    plt.plot(lstm_pred_df.index.to_timestamp(), lstm_pred_df['cnt'], label='Actual')
    plt.plot(lstm_pred_df.index.to_timestamp(), lstm_pred_df['predictions'], label='Predictions')
    plt.plot(test_df.head(lookback).index.to_timestamp(), test_df.head(lookback)['cnt'], label='Test Lookback Period')
    plt.title('Bike Rentals')
    plt.xlabel('Date')
    plt.ylabel('Number of Rentals')
    plt.title('Actual vs Predicted for LSTM')
    plt.legend()
    plt.show()