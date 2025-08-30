import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
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

# Build neural network model
model = Sequential([
    Dense(256, activation='relu', input_shape=(X.shape[1],)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(1)
])

# Compile model, adding 'mae' as a metric
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='mean_squared_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Early stopping callback
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train model
start_time = time.time()
history = model.fit(X_train, y_train,
          validation_data=(X_val, y_val),
          epochs=100,
          batch_size=1024,
          callbacks=[early_stop],
          verbose=1)
training_time = time.time() - start_time


# Since restore_best_weights is True, find the metrics from the epoch with the best validation loss
best_epoch = np.argmin(history.history['val_loss'])
best_val_loss = history.history['val_loss'][best_epoch]
best_train_loss = history.history['loss'][best_epoch]
best_val_mae = history.history['val_rmse'][best_epoch]
best_train_mae = history.history['rmse'][best_epoch]


# Create results dictionary
results = {
    'best_validation_loss (rmse)': best_val_loss,
    'best_training_loss (rmse)': best_train_loss,
    'best_validation_rmse': best_val_mae,
    'best_training_rmse': best_train_mae,
    'training_time_seconds': training_time
}

# Save results to a JSON file
with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nTraining complete in {training_time:.2f} seconds.")
print(f"Best Validation MAE: {best_val_mae}")
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
