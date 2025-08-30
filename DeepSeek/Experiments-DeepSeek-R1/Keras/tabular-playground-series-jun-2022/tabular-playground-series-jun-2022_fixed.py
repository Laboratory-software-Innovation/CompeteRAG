import pandas as pd
import numpy as np
import time
import json
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
# Load dataset
# Note: You will need to adjust the file path to where you have stored the data.
# For this example, it assumes the data is in a 'test/tabular-playground-series-jun-2022/tabular-playground-series-jun-2022/' directory.
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

# Build model
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(processed_data.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(continuous_cols))
])
model.compile(optimizer='adam', loss='mse', 
metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Train model with timing
start_time = time.time()
history = model.fit(
    processed_data, 
    data_continuous_scaled.values, 
    epochs=100, 
    batch_size=1024, 
    validation_split=0.2, 
    verbose=1
)
training_time = time.time() - start_time

# Save training results, including the training time
results = {
    'training_rmse': history.history['rmse'][-1],
    'training_loss': history.history['loss'][-1],
    'validation_rmse': history.history['val_rmse'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'training_time': training_time  # Added training time here
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete. Time taken: {training_time:.2f} seconds.")
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
