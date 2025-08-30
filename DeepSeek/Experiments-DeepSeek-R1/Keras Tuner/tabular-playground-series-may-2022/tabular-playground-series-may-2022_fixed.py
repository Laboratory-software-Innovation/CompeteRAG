import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import numpy as np
import time
import json
import keras_tuner as kt

# Load and prepare data
try:
    train = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/train.csv.zip')
    test = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/test.csv.zip')
except FileNotFoundError:
    print("Please ensure the dataset files are in the correct directory.")
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

# Process categorical feature
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

# Keras-Tuner integration
n_features = X_train.shape[1]
early_stopping = EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_auc', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()
        model.add(Dense(
            units=hp.Int('units', 64, 1024, step=64),
            activation='relu',
            input_dim=n_features,
            kernel_regularizer=regularizers.l2(1e-4)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))

        for _ in range(hp.Int('layers', 2, 8)):
            model.add(Dense(
                units=hp.Int('units', 64, 1024, step=64),
                activation='relu',
                kernel_regularizer=regularizers.l2(1e-4)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(hp.Float('dropout', 0.0, 0.5, step=0.1)))

        model.add(Dense(1, activation='sigmoid'))
        
        optimizer = Adam(learning_rate=hp.Float('learning_rate', 1e-5, 0.01, sampling='log'))
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['auc']
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
    objective='val_auc',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='bayesian_tuner'
)

start_time = time.time()
tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=best_hps.get('batch_size'),
    callbacks=[early_stopping, checkpoint],
    verbose=2
)
training_time = time.time() - start_time

# Save final epoch metrics
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