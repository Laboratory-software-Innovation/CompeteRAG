import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError # Import RMSE metric

# Load and preprocess data
try:
    train = pd.read_csv('test/tabular-playground-series-aug-2021/tabular-playground-series-aug-2021/train.csv.zip')
    test = pd.read_csv('test/tabular-playground-series-aug-2021/tabular-playground-series-aug-2021/test.csv.zip')
except FileNotFoundError:
    print("Ensure 'train.csv' and 'test.csv' are in the correct directory.")
    exit()

# Separate features and target
X = train.drop(['id', 'loss'], axis=1)
y = train['loss'].values.reshape(-1, 1)
test_ids = test['id']
test = test.drop('id', axis=1)

# Handle missing values
imputer = SimpleImputer(strategy='median')
X = imputer.fit_transform(X)
test = imputer.transform(test)

# Feature scaling
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
test = scaler.transform(test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ========== Keras-Tuner Integration Start ==========
import keras_tuner as kt

n_features = X_train.shape[1]
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Input(shape=(n_features,)))
        
        layers = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 1024, step=64)
        drop = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')

        for _ in range(layers):
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(drop))
        
        model.add(Dense(1))
        
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss='mean_squared_error',
            metrics=[RootMeanSquaredError()] # Keras tracks loss (MSE) by default
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            epochs=hp.Int('epochs', 20, 200, step=10),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_root_mean_squared_error', # Still optimizing for RMSE
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='insurance_tuner_rmse'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)
# =========== Keras-Tuner Integration End ===========

# Train the best model found by the tuner
start_time = time.time()
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=best_hps.get('batch_size'),
                    callbacks=[early_stop],
                    verbose=1)
training_time = time.time() - start_time

# --- MODIFIED RESULTS COLLECTION ---
# Find the metrics from the epoch with the best validation RMSE
best_epoch = np.argmin(history.history['val_root_mean_squared_error'])

# Get RMSE scores from that epoch
best_val_rmse = history.history['val_root_mean_squared_error'][best_epoch]
best_train_rmse = history.history['root_mean_squared_error'][best_epoch]

# Get loss (MSE) scores from that same best epoch
best_val_loss_mse = history.history['val_loss'][best_epoch]
best_train_loss_mse = history.history['loss'][best_epoch]


# Create results dictionary with both RMSE and the original loss (MSE)
results = {
    'best_validation_rmse': best_val_rmse,
    'best_training_rmse': best_train_rmse,
    'best_validation_loss_mse': best_val_loss_mse,
    'best_training_loss_mse': best_train_loss_mse,
    'training_time_seconds': training_time
}

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Best Validation Loss (MSE): {best_val_loss_mse:.4f}")
print(f"Best Validation RMSE: {best_val_rmse:.4f}")
print("Results saved to results.json")

# Generate predictions
test_pred = model.predict(test).flatten()

# Create submission file
submission = pd.DataFrame({
    'id': test_ids,
    'loss': test_pred
})
submission.to_csv('submission.csv', index=False)
print("Submission file created.")