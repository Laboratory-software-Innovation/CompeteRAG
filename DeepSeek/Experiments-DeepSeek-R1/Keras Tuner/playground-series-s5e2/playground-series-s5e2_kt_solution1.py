<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load and merge training data
train = pd.read_csv('test/playground-series-s5e2/playground-series-s5e2/train.csv')
train_extra = pd.read_csv('test/playground-series-s5e2/playground-series-s5e2/training_extra.csv')
data = pd.concat([train, train_extra], axis=0)

# Define columns
target_col = 'Price'
id_col = 'id'
numerical_cols = ['Compartments', 'Weight Capacity (kg)']
cat_cols = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

# Preprocessing configuration
num_medians = data[numerical_cols].median().to_dict()
label_encoders = {}

# Handle missing values and encode categorical features
for col in cat_cols:
    data[col] = data[col].fillna('Missing')
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Process numerical features
scaler = StandardScaler()
data[numerical_cols] = scaler.fit_transform(data[numerical_cols].fillna(num_medians))

# Split training data
X = data.drop([target_col, id_col], axis=1)
y = data[target_col]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)

# Keras-Tuner implementation starts here
# Define early stopping and model checkpoint callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Get input dimension
n_features = X_train.shape[1]

import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        # Hyperparameters from chosen bank
        layers = hp.Int('layers', min_value=2, max_value=8, step=1)
        units = hp.Int('units', min_value=64, max_value=1024, step=64)
        activation = hp.Choice('activation', ['relu'])
        dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        optimizer = hp.Choice('optimizer', ['adam'])
        learning_rate = hp.Float('learning_rate', min_value=1e-05, max_value=0.01, sampling='log')

        # Model architecture
        inputs = Input(shape=(n_features,))
        x = inputs
        
        # Add tunable dense layers
        for _ in range(layers):
            x = Dense(units, activation=activation)(x)
            x = Dropout(dropout)(x)
            
        output = Dense(1, activation='linear')(x)
        model = Model(inputs, output)
        
        # Compile with tunable parameters
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae'],
            learning_rate=learning_rate
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            epochs=hp.Int('epochs', min_value=20, max_value=200, step=10),
            **kwargs
        )

# Initialize Bayesian tuner
tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='backpack_pricing_tuner'
)

# Execute search
tuner.search(
    X_train.values, y_train,
    validation_data=(X_val.values, y_val),
    callbacks=[early_stopping, checkpoint]
)

# Retrieve best model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Final training with full epochs
history = model.fit(
    X_train.values, y_train,
    validation_data=(X_val.values, y_val),
    epochs=200,
    batch_size=best_hps.get('batch_size'),
    callbacks=[early_stopping, checkpoint],
    verbose=2
)

# Prepare input data
train_cat = [X_train[col].values for col in cat_cols]
train_num = X_train[numerical_cols].values
val_cat = [X_val[col].values for col in cat_cols]
val_num = X_val[numerical_cols].values

# Train model
model.fit(
    train_cat + [train_num], y_train,
    validation_data=(val_cat + [val_num], y_val),
    epochs=100,
    batch_size=1024,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

# Process test data
test = pd.read_csv('test/playground-series-s5e2/playground-series-s5e2/test.csv')
test_ids = test[id_col]

# Preprocess categorical test data
test_cat_data = test[cat_cols].copy()
for col in cat_cols:
    test_cat_data[col] = test_cat_data[col].fillna('Missing')
    valid_labels = label_encoders[col].classes_
    mask = ~test_cat_data[col].isin(valid_labels)
    test_cat_data.loc[mask, col] = 'Missing'
    test_cat_data[col] = label_encoders[col].transform(test_cat_data[col])

# Preprocess numerical test data
test_num_data = test[numerical_cols].copy()
for col in numerical_cols:
    test_num_data[col] = test_num_data[col].fillna(num_medians[col])
test_num_data = scaler.transform(test_num_data)

# Generate predictions
test_cat_list = [test_cat_data[col].values for col in cat_cols]
preds = model.predict(test_cat_list + [test_num_data])

# Create submission
submission = pd.DataFrame({id_col: test_ids, target_col: preds.flatten()})
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s5e2/playground-series-s5e2_kt_solution.py", line 95, in <module>
    tuner = kt.BayesianOptimization(
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/tuners/bayesian.py", line 394, in __init__
    super().__init__(oracle=oracle, hypermodel=hypermodel, **kwargs)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/tuner.py", line 122, in __init__
    super().__init__(
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 132, in __init__
    self._populate_initial_space()
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 192, in _populate_initial_space
    self._activate_all_conditions()
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/base_tuner.py", line 149, in _activate_all_conditions   
    self.hypermodel.build(hp)
  File "/usr/local/lib/python3.10/dist-packages/keras_tuner/src/engine/hypermodel.py", line 120, in _build_wrapper
    return self._build(hp, *args, **kwargs)
  File "/app/test/playground-series-s5e2/playground-series-s5e2_kt_solution.py", line 78, in build
    model.compile(
  File "/usr/local/lib/python3.10/dist-packages/keras/src/utils/traceback_utils.py", line 122, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "/usr/local/lib/python3.10/dist-packages/keras/src/utils/tracking.py", line 26, in wrapper
    return fn(*args, **kwargs)
TypeError: Trainer.compile() got an unexpected keyword argument 'learning_rate'
</Error>