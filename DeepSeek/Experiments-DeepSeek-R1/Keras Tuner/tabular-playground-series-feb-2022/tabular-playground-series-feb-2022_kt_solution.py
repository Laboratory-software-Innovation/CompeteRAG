
import pandas as pd
import numpy as np
import json
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

# Load training data with Python engine to handle malformed CSV
train_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/train.csv.zip', 
                      encoding='latin-1', 
                      engine='python')
X = train_df.drop(['row_id', 'target'], axis=1).values
y = train_df['target']

# Encode labels
encoder = LabelEncoder()
encoded_y = encoder.fit_transform(y)
y_categorical = to_categorical(encoded_y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42)

# Keras-Tuner hypermodel setup
n_features = X_train.shape[1]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', 2, 6, step=1)
        units = hp.Int('units', 64, 512, step=64)
        act = hp.Choice('activation', ['relu', 'tanh', 'selu'])
        drop = hp.Float('dropout', 0.0, 0.5, step=0.1)
        opt = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('learning_rate', 0.0001, 0.01, sampling='log')
        bs = hp.Choice('batch_size', [64, 128, 256, 512])
        ep = hp.Int('epochs', 10, 50, step=5)

        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation=act, kernel_initializer='he_normal')(x)
            x = BatchNormalization()(x)
            x = Dropout(drop)(x)
        outputs = Dense(y_categorical.shape[1], activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(
            optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
            batch_size=hp.get('batch_size'),
            epochs=hp.get('epochs'),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='bacteria_tuner'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# Train model with time tracking
start_time = time.time()
history = model.fit(X_train, y_train,
                    epochs=30,
                    batch_size=1024,
                    validation_data=(X_val, y_val),
                    verbose=1)
training_time = time.time() - start_time

# Load test data with Python engine
test_df = pd.read_csv('test/tabular-playground-series-feb-2022/tabular-playground-series-feb-2022/test.csv.zip', 
                     encoding='latin-1',
                     engine='python')
X_test = scaler.transform(test_df.drop(['row_id'], axis=1).values)

# Generate predictions
probabilities = model.predict(X_test)
predicted_labels = encoder.inverse_transform(np.argmax(probabilities, axis=1))

# Create submission
submission = pd.DataFrame({
    'row_id': test_df['row_id'],
    'target': predicted_labels
})
submission.to_csv('submission.csv', index=False)

# Save training metrics
results = {
    'training_accuracy': history.history['accuracy'][-1],
    'training_loss': history.history['loss'][-1],
    'validation_accuracy': history.history['val_accuracy'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'training_time_seconds': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)
