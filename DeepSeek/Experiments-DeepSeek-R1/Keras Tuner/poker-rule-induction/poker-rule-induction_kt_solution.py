<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import time
import json
import keras_tuner as kt

# Load and prepare data
train = pd.read_csv('test/poker-rule-induction/poker-rule-induction/train.csv.zip')
X = train.drop('hand', axis=1).values
y = train['hand'].values.astype(int)

# Create ordinal targets for CORAL method
y_coral = np.zeros((len(y), 9))
for i in range(9):
    y_coral[:, i] = (y > i).astype(float)

# Split and scale data
X_train, X_val, y_train, y_val = train_test_split(X, y_coral, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Keras-Tuner hypermodel definition
n_features = X_train.shape[1]

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', 2, 6)
        units = hp.Int('units', 64, 512, step=64)
        act = hp.Choice('activation', ['relu', 'tanh', 'selu'])
        drop = hp.Float('dropout', 0.0, 0.5, step=0.1)
        opt = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('learning_rate', 0.0001, 0.01, sampling='log')

        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation=act)(x)
            x = Dropout(drop)(x)
        outputs = Dense(10, activation='softmax')(x)
        model = Model(inputs, outputs)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(*args,
            batch_size=hp.Choice('batch_size', [64, 128, 256, 512]),
            epochs=hp.Int('epochs', 10, 50, step=5),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    project_name='poker_tuner'
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_poker_model.h5', monitor='val_loss', save_best_only=True
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)

# Train with early stopping
es = EarlyStopping(patience=5, restore_best_weights=True)

start_time = time.time()
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=256,
                    callbacks=[es],
                    class_weight={i: 1.0 for i in range(9)})
end_time = time.time()

# Save training results
training_time = end_time - start_time
best_epoch = np.argmin(history.history['val_loss'])

results = {
    'training_accuracy': history.history['accuracy'][best_epoch],
    'training_loss': history.history['loss'][best_epoch],
    'validation_accuracy': history.history['val_accuracy'][best_epoch],
    'validation_loss': history.history['val_loss'][best_epoch],
    'training_time': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("\nTraining results saved to results.json")

# Prepare test predictions - FIX: Test data has 'id' column that needs to be dropped
test_df = pd.read_csv('test/poker-rule-induction/poker-rule-induction/test.csv.zip')
X_test = scaler.transform(test_df.drop('id', axis=1).values)  # Drop ID column before scaling
preds = model.predict(X_test)
test_predictions = np.sum(preds >= 0.5, axis=1)

# Save submission
submission = pd.read_csv('test/poker-rule-induction/poker-rule-induction/sampleSubmission.csv.zip')
submission['hand'] = test_predictions.astype(int)
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 274, in _try_run_and_update_trial
    self._run_and_update_trial(trial, *fit_args, **fit_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 239, in _run_and_update_trial
    results = self.run_trial(trial, *fit_args, **fit_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/tuner.py", line 314, in run_trial
    obj_value = self._build_and_fit_model(trial, *args, **copied_kwargs)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/tuner.py", line 233, in _build_and_fit_model
    results = self.hypermodel.fit(hp, model, *args, **kwargs)
  File "/app/test/poker-rule-induction/poker-rule-induction_kt_solution.py", line 53, in fit
    return model.fit(*args,
  File "/usr/local/lib/python3.10/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.10/dist-packages/keras/src/backend/tensorflow/nn.py", line 783, in binary_crossentropy
    raise ValueError(
ValueError: Arguments `target` and `output` must have the same shape. Received: target.shape=(None, 9), output.shape=(None, 10)
</Error>