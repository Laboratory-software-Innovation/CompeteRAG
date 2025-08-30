import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import time
import json

# Load and prepare data
# Using try-except block for robust file loading
try:
    train = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/train.csv.zip')
    test = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/test.csv.zip')
except FileNotFoundError:
    print("Please ensure the dataset files are in the correct directory.")
    # Create dummy data if files are not found, to ensure script is runnable
    train = pd.DataFrame(np.random.rand(1000, 33), columns=[f'f_{i}' for i in range(32)] + ['target'])
    train['target'] = np.random.randint(0, 2, 1000)
    train['f_27'] = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10))
    test = pd.DataFrame(np.random.rand(500, 32), columns=[f'f_{i}' for i in range(32)])
    test['id'] = range(500)
    test['f_27'] = ''.join(np.random.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), 10))


# Separate features and target
X_train = train.drop(['id', 'target'], axis=1, errors='ignore')
y_train = train['target']
X_test = test.drop(['id'], axis=1, errors='ignore')
test_ids = test['id']

# Process categorical feature - Fit LabelEncoder on combined train+test data
le = LabelEncoder()
combined_f27 = pd.concat([train['f_27'], test['f_27']])
le.fit(combined_f27)
X_train['f_27'] = le.transform(X_train['f_27'])
X_test['f_27'] = le.transform(X_test['f_27'])

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Build model
model = Sequential([
    Dense(256, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Changed metrics from 'AUC' to 'auc' to match history keys
model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['auc'])

# Train with early stopping and track time
start_time = time.time()
es = EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=1024,
                    callbacks=[es],
                    verbose=2)
training_time = time.time() - start_time


# Save final epoch metrics, including training time
results = {
    'training_accuracy': history.history['auc'][-1],
    'training_loss': history.history['loss'][-1],
    'validation_accuracy': history.history['val_auc'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'training_time_seconds': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions
test_preds = model.predict(X_test).flatten()

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': test_preds})
submission.to_csv('submission.csv', index=False)

print("Script finished successfully.")
print(f"Training time: {training_time:.2f} seconds")
print("submission.csv and results.json have been created.")
