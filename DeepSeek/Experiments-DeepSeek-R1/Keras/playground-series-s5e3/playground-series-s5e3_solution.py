<Code>
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers, callbacks

# Load data
train_df = pd.read_csv('/media/volume/kaggled/GitHub/Version 1/REU/test/playground-series-s5e3/playground-series-s5e3/train.csv')
test_df = pd.read_csv('/media/volume/kaggled/GitHub/Version 1/REU/test/playground-series-s5e3/playground-series-s5e3/test.csv')

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
class_weights = {
    0: (1 / np.mean(y_train == 0)) * 0.5,
    1: (1 / np.mean(y_train == 1)) * 0.5
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
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[keras.metrics.AUC(name='auc')]
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    patience=10, restore_best_weights=True, monitor='val_auc', mode='max'
)

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

# Generate predictions
test_preds = model.predict(X_test).flatten()

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'rainfall': test_preds
})
submission.to_csv('submission.csv', index=False)
</Code>


<Error>
When submitting the submission file I got this error:
'Submission contains null values'
</Error>
