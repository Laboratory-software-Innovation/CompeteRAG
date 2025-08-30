import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import time
import numpy as np
import json

# Load and prepare data
train_df = pd.read_csv('test/playground-series-s5e5/playground-series-s5e5/train.csv')
test_df = pd.read_csv('test/playground-series-s5e5/playground-series-s5e5/test.csv')
test_ids = test_df['id']

# Feature/target separation
X = train_df.drop(['id', 'Calories'], axis=1)
y = train_df['Calories']
X_test = test_df.drop('id', axis=1)

# Feature engineering
num_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
cat_features = ['Sex']

# Missing value imputation
imputer = SimpleImputer(strategy='median')
X[num_features] = imputer.fit_transform(X[num_features])
X_test[num_features] = imputer.transform(X_test[num_features])

# Categorical encoding
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex'])
X_test['Sex'] = le.transform(X_test['Sex'])

# Feature scaling
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Neural network architecture
model = Sequential([
    Dense(512, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(1)
])

# Model configuration
model.compile(optimizer='adam', loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])

# Train/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Start timer
start_time = time.time()

# Training with early stopping
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=1024,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=0
)

# End timer and calculate duration
training_time = time.time() - start_time

# --- CHANGED: Section to save metrics to a file ---
# Find the best epoch (since restore_best_weights=True is used)
best_epoch = np.argmin(history.history['val_loss'])

# Create a dictionary with the results from the best epoch in the desired format
results = {
    "training_rmse": float(round(history.history['root_mean_squared_error'][best_epoch], 4)),
    "training_loss": float(round(history.history['loss'][best_epoch], 4)),
    "validation_rmse": float(round(history.history['val_root_mean_squared_error'][best_epoch], 4)),
    "validation_loss": float(round(history.history['val_loss'][best_epoch], 4)),
    "training_time_seconds": round(training_time, 2)
}

# Save the results dictionary to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

# --- END OF CHANGE ---

# Generate predictions
predictions = model.predict(X_test).flatten()

# Create submission file
pd.DataFrame({'id': test_ids, 'Calories': predictions}).to_csv('sample_submission.csv', index=False)

print("Submission file 'sample_submission.csv' and 'results.json' created successfully.")
