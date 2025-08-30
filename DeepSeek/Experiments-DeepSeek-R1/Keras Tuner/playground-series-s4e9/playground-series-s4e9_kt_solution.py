<Code>
import pandas as pd
import numpy as np
import re
import json
import time
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Load Data
try:
    train = pd.read_csv('test/playground-series-s4e9/playground-series-s4e9/train.csv.zip')
    test = pd.read_csv('test/playground-series-s4e9/playground-series-s4e9/test.csv.zip')
except FileNotFoundError:
    train = pd.read_csv('train.csv.zip')
    test = pd.read_csv('test.csv.zip')

# Feature Engineering & Preprocessing
def parse_engine(s):
    if not isinstance(s, str):
        return np.nan, np.nan
    displacement_match = re.search(r'(\d+\.\d+)', s)
    cylinders_match = re.search(r'V(\d+)', s)
    displacement = float(displacement_match.group(1)) if displacement_match else np.nan
    cylinders = int(cylinders_match.group(1)) if cylinders_match else np.nan
    return displacement, cylinders

for df in [train, test]:
    engine_features = df['engine'].apply(parse_engine)
    df[['displacement', 'cylinders']] = pd.DataFrame(engine_features.tolist(), index=df.index)

# Use direct assignment for fillna to avoid 'SettingWithCopyWarning' and ensure NaNs are filled.
for col in ['fuel_type', 'accident', 'clean_title']:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)

numerical_cols_to_impute = ['model_year', 'milage', 'displacement', 'cylinders']
for col in numerical_cols_to_impute:
    median_val = train[col].median()
    train[col] = train[col].fillna(median_val)
    test[col] = test[col].fillna(median_val)

# This converts 'Yes' to 1 and everything else (including 'No') to 0.
train['accident'] = (train['accident'] == 'Yes').astype(int)
test['accident'] = (test['accident'] == 'Yes').astype(int)
train['clean_title'] = (train['clean_title'] == 'Yes').astype(int)
test['clean_title'] = (test['clean_title'] == 'Yes').astype(int)


categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col] = le.transform(test[col].astype(str))
    label_encoders[col] = le

# Prepare Data for Model
numerical_cols = ['model_year', 'milage', 'displacement', 'cylinders', 'accident', 'clean_title']

# NEW DEBUG STEP: Check for NaNs in the dataframe before converting to numpy
print("\n--- Checking for NaNs in DataFrame before splitting ---")
for col in numerical_cols:
    nan_count = train[col].isnull().sum()
    if nan_count > 0:
        print(f"WARNING: Found {nan_count} NaN(s) in column '{col}' after preprocessing.")
print("-----------------------------------------------------\n")


X_numerical = train[numerical_cols].values
X_categorical = [train[col].values for col in categorical_cols]
y = train['price'].values

# Use a single, correct train_test_split call and unpack results cleanly.
all_data_to_split = [X_numerical] + X_categorical + [y]
split_results = train_test_split(*all_data_to_split, test_size=0.2, random_state=42)

# Unpack the results correctly with proper reshaping for categorical features
X_num_train = split_results[0]
X_num_val = split_results[1]
# Reshape categorical features to 2D arrays
X_cat_train = [split_results[i].reshape(-1, 1) for i in range(2, 2 + 2*len(categorical_cols), 2)]
X_cat_val = [split_results[i].reshape(-1, 1) for i in range(3, 3 + 2*len(categorical_cols), 2)]
y_train = split_results[-2]
y_val = split_results[-1]

scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_val = scaler.transform(X_num_val)

# Build Model with Keras Tuner
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import ModelCheckpoint

# Preserve original callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

n_features = X_num_train.shape[1] + sum([x.shape[1] for x in X_cat_train])

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', 2, 8)
        units = hp.Int('units', 64, 1024, step=64)
        act = hp.Choice('activation', ['relu', 'tanh', 'selu'])
        drop = hp.Float('dropout', 0.0, 0.5)
        opt = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        
        # Recreate original model structure with hyperparameters
        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation=act)(x)
            x = Dropout(drop)(x)
        output = Dense(1)(x)
        
        # Configure optimizer with gradient clipping
        if opt == 'adam':
            optimizer = Adam(learning_rate=lr, clipnorm=1.0)
        elif opt == 'rmsprop':
            optimizer = RMSprop(learning_rate=lr, clipnorm=1.0)
        else:
            optimizer = SGD(learning_rate=lr, clipnorm=1.0)
        
        model = Model(inputs, output)
        model.compile(optimizer=optimizer, loss='mse')
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
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='car_price_tuner'
)

# Convert multiple inputs to single array for tuning
X_train_proc = np.hstack([X_num_train] + [x.astype(np.float32) for x in X_cat_train])
X_val_proc = np.hstack([X_num_val] + [x.astype(np.float32) for x in X_cat_val])

tuner.search(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    callbacks=[early_stopping, checkpoint]
)

# Retrain best model
best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

start_time = time.time()
history = model.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=best_hps.get('batch_size'),
    callbacks=[early_stopping, checkpoint],
    verbose=2
)
training_time = time.time() - start_time

# Train Model
X_train_inputs = [X_num_train] + X_cat_train
X_val_inputs = [X_num_val] + X_cat_val

# FINAL SANITY CHECK
print("\n--- Data Check Before Training ---")
assert not np.isnan(X_num_train).any(), "FATAL: NaNs found in numerical training features!"
assert not any(np.isnan(cat).any() for cat in X_cat_train), "FATAL: NaNs found in categorical training features!"
assert not np.isnan(y_train).any(), "FATAL: NaNs found in the target variable (y_train)!"
print("Data is clean. Starting training...")
print("---------------------------------\n")

# Save Results and Predict
if history and history.history:
    last_epoch = len(history.history['loss']) - 1
    results = {
        'training_loss': history.history['loss'][last_epoch],
        'validation_loss': history.history['val_loss'][last_epoch],
        'training_time': training_time
    }
    with open('results.json', 'w') as f:
        json.dump(results, f)

test_num = scaler.transform(test[numerical_cols].values)
test_cat = [test[col].values.reshape(-1, 1) for col in categorical_cols]
test_inputs = [test_num] + test_cat
predictions = model.predict(test_inputs).flatten()

assert not np.isnan(predictions).any(), "Predictions contain NaN values"

submission = pd.DataFrame({'id': test['id'], 'price': predictions})
submission.to_csv('submission.csv', index=False)

print("\nScript finished successfully. Submission file created.")
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s4e9/playground-series-s4e9_fixed.py", line 208, in <module>
    predictions = model.predict(test_inputs).flatten()
  File "/usr/local/lib/python3.10/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.10/dist-packages/keras/src/layers/input_spec.py", line 160, in assert_input_compatibility
    raise ValueError(
ValueError: Layer "functional_1" expects 1 input(s), but it received 7 input tensors. Inputs received: [<tf.Tensor 'data:0' shape=(32, 6) dtype=float32>, <tf.Tensor 'data_1:0' shape=(32, 1) dtype=int64>, <tf.Tensor 'data_2:0' shape=(32, 1) dtype=int64>, <tf.Tensor 'data_3:0' shape=(32, 1) dtype=int64>, <tf.Tensor 'data_4:0' shape=(32, 1) dtype=int64>, <tf.Tensor 'data_5:0' shape=(32, 1) dtype=int64>, <tf.Tensor 'data_6:0' shape=(32, 1) dtype=int64>]
</Error>