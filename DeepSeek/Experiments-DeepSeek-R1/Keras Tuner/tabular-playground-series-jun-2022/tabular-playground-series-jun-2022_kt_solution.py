import pandas as pd
import numpy as np
import time
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import RootMeanSquaredError
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Load dataset
try:
    data = pd.read_csv('test/tabular-playground-series-jun-2022/tabular-playground-series-jun-2022/data.csv.zip')
except FileNotFoundError:
    print("Please download the dataset from Kaggle 'Tabular Playground Series - Jun 2022' and place data.csv.zip in the correct directory.")
    exit()


# Identify continuous and categorical columns
continuous_cols = data.columns[data.columns.str.startswith(('F_1_', 'F_3_', 'F_4_'))].tolist()
categorical_cols = data.columns[data.columns.str.startswith('F_2_')].tolist()
all_continuous = data[continuous_cols].copy()

# Create mask for original missing values
missing_mask = data[continuous_cols].isna()

# Impute missing continuous values with median
imputer = SimpleImputer(strategy='median')
data_continuous = imputer.fit_transform(data[continuous_cols])
data_continuous = pd.DataFrame(data_continuous, columns=continuous_cols)

# Standardize continuous features
scaler = StandardScaler()
data_continuous_scaled = scaler.fit_transform(data_continuous)
data_continuous_scaled = pd.DataFrame(data_continuous_scaled, columns=continuous_cols)

# One-hot encode categorical features
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
data_categorical = ohe.fit_transform(data[categorical_cols])

# Combine processed features
processed_data = np.concatenate([data_continuous_scaled, data_categorical], axis=1)

# Build and tune model with Keras Tuner
n_features = processed_data.shape[1]
target_data = data_continuous_scaled.values
expected_early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        hp_layers = hp.Int('layers', 2, 8, step=1)
        hp_units = hp.Int('units', 64, 1024, step=64)
        hp_dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        hp_learning_rate = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')

        model = keras.Sequential()
        model.add(layers.Input(shape=(n_features,)))
        
        for _ in range(hp_layers):
            model.add(layers.Dense(hp_units, activation='relu'))
            model.add(layers.Dropout(hp_dropout))
        
        model.add(layers.Dense(len(continuous_cols)))
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='mse',
            metrics=[RootMeanSquaredError()]
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
    objective='val_root_mean_squared_error',
    max_trials=10,
    executions_per_trial=1,
    directory='keras_tuner',
    project_name='imputation_tuning_rmse'
)

tuner.search(
    processed_data,
    target_data,
    validation_split=0.2,
    callbacks=[expected_early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)

# Train model with timing
start_time = time.time()
history = model.fit(
    processed_data, 
    target_data, 
    epochs=100,
    batch_size=best_hps.get('batch_size'),
    validation_split=0.2,
    callbacks=[expected_early_stopping, checkpoint],
    verbose=1
)
training_time = time.time() - start_time

# --- MODIFIED: Save both loss (MSE) and RMSE results ---
# Find the best epoch based on the validation RMSE
best_epoch = np.argmin(history.history['val_root_mean_squared_error'])
training_rmse = history.history['root_mean_squared_error'][best_epoch]
validation_rmse = history.history['val_root_mean_squared_error'][best_epoch]

# Get the corresponding loss (MSE) values from that same best epoch
training_loss = history.history['loss'][best_epoch]
validation_loss = history.history['val_loss'][best_epoch]

# Create results dictionary with both sets of metrics
results = {
    'training_rmse': training_rmse,
    'validation_rmse': validation_rmse,
    'training_loss_mse': training_loss,
    'validation_loss_mse': validation_loss,
    'training_time': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete. Time taken: {training_time:.2f} seconds.")
print(f"Best Validation Loss (MSE): {validation_loss:.4f}")
print(f"Best Validation RMSE: {validation_rmse:.4f}")
print("Training metrics and time saved to results.json")

# Generate predictions for all continuous features
predictions = model.predict(processed_data)
predicted_continuous = scaler.inverse_transform(predictions)

# Prepare submission
submission_list = []
for row_idx in range(data.shape[0]):
    for col_idx, col in enumerate(continuous_cols):
        if missing_mask.iloc[row_idx, col_idx]:
            submission_list.append({
                'row-col': f"{row_idx}-{col}",
                'value': predicted_continuous[row_idx, col_idx]
            })

# Create and save submission
pd.DataFrame(submission_list).to_csv('submission.csv', index=False)
print("Submission file created as submission.csv")