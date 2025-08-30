import pandas as pd
import numpy as np
import subprocess
import sys

# Install category_encoders if missing
try:
    from category_encoders import CatBoostEncoder
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "category-encoders"])
    from category_encoders import CatBoostEncoder

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import keras_tuner as kt
import time
import json

# Start timer
start_time = time.time()

# Load data
train = pd.read_csv('test/playground-series-s4e10/playground-series-s4e10/train.csv')
test = pd.read_csv('test/playground-series-s4e10/playground-series-s4e10/test.csv')

# Prepare features and target
X = train.drop(['id', 'loan_status'], axis=1)
y = train['loan_status']
X_test = test.drop('id', axis=1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical columns
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
num_cols = [col for col in X_train.columns if col not in cat_cols]

# Encode categorical features
encoder = CatBoostEncoder()
X_train_encoded = encoder.fit_transform(X_train[cat_cols], y_train)
X_val_encoded = encoder.transform(X_val[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

# Merge encoded features with numerical features
X_train = pd.concat([X_train[num_cols], X_train_encoded], axis=1)
X_val = pd.concat([X_val[num_cols], X_val_encoded], axis=1)
X_test = pd.concat([X_test[num_cols], X_test_encoded], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Hypermodel definition
n_features = X_train.shape[1]

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Hyperparameters
        layers = hp.Int('layers', 2, 6)
        units = hp.Int('units', 64, 512, step=64)
        activation = hp.Choice('activation', ['relu', 'tanh', 'selu'])
        dropout = hp.Float('dropout', 0.0, 0.5)
        optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
        
        # Learning rate schedule
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr,
            decay_steps=3200,
            decay_rate=0.9,
            staircase=True
        )
        
        # Model architecture
        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation=activation)(x)
            x = Dropout(dropout)(x)
        outputs = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, outputs)
        
        # Optimizer configuration
        if optimizer == 'adam':
            opt = Adam(learning_rate=lr_schedule)
        elif optimizer == 'rmsprop':
            opt = RMSprop(learning_rate=lr_schedule)
        else:
            opt = SGD(learning_rate=lr_schedule)
        
        model.compile(loss='binary_crossentropy',
                      optimizer=opt,
                      metrics=['accuracy'])
        return model

# Tuner setup
tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=False,
    project_name='loan_approval_tuner'
)

# Early stopping and checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Hyperparameter search
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    batch_size=64,
    epochs=50,
    callbacks=[early_stopping, checkpoint],
    verbose=1
)

# Retrain best model
best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,  # Fixed batch size since it's not part of hyperparameter search
    callbacks=[early_stopping, checkpoint],
    verbose=2
)

# Save results
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history['accuracy'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_accuracy'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time_seconds': time.time() - start_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions
test_pred = model.predict(X_test).flatten()

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'loan_status': test_pred})
submission.to_csv('submission.csv', index=False)