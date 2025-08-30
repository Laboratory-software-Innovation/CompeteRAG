import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import time

# Load data
train_df = pd.read_csv('test/playground-series-s5e3/playground-series-s5e3/train.csv')
test_df = pd.read_csv('test/playground-series-s5e3/playground-series-s5e3/test.csv')

# Preprocessing
X = train_df.drop(columns=['id', 'rainfall', 'day'])
y = train_df['rainfall'].values
X_test = test_df.drop(columns=['id', 'day'])

# Split train/validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Class weights for imbalance
class_counts = np.bincount(y_train.astype(int))
total_samples = len(y_train)
class_weights = {
    0: total_samples / (2 * class_counts[0]) if class_counts[0] > 0 else 1.0,
    1: total_samples / (2 * class_counts[1]) if class_counts[1] > 0 else 1.0
}

# Build model
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(name='auc')]
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, monitor='val_auc', mode='max'
)

# Track training time
start_time = time.time()

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop],
    verbose=2
)

training_time = time.time() - start_time

# Generate predictions (ensure no NaNs)
test_preds = model.predict(X_test, verbose=0).flatten()
test_preds = np.nan_to_num(test_preds, nan=0.5)  # Replace NaNs with 0.5

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds
})
submission.to_csv('submission.csv', index=False)

# Save training results
import json
last_epoch = len(history.history['auc']) - 1
results = {
    'training_accuracy': history.history['auc'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_auc'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)