<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
import time
import json

# Load data
train = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/train.csv.zip')
test = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/test.csv.zip')
submission = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/sample_submission.csv.zip')

# Prepare data
y_train = train['target']
id_test = test['id']

# Categorical columns setup
binary_cols = ['bin_3', 'bin_4']
numerical_cols = ['bin_0', 'bin_1', 'bin_2', 'ord_0']
ordinal_cols = ['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
nominal_cols = [f'nom_{i}' for i in range(10)]
cyclical_cols = ['day', 'month']

# Handle missing values
train[cyclical_cols] = train[cyclical_cols].fillna(train[cyclical_cols].median())
test[cyclical_cols] = test[cyclical_cols].fillna(test[cyclical_cols].median())

imputer = SimpleImputer(strategy='median')
train[numerical_cols] = imputer.fit_transform(train[numerical_cols])
test[numerical_cols] = imputer.transform(test[numerical_cols])

for col in binary_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val).map({mode_val: 0, list(train[col].unique())[1 - list(train[col].unique()).index(mode_val)]: 1})
    test[col] = test[col].fillna(test[col].mode()[0]).map({mode_val: 0, list(test[col].unique())[1 - list(test[col].unique()).index(mode_val)]: 1})

# Process cyclical features
for df in [train, test]:
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

cyclical_transformed = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

# Process ordinal features - FIXED SECTION
ordinal_maps = {}
for col in ordinal_cols:
    # Handle missing values first
    train[col] = train[col].fillna('missing')
    unique_vals = sorted(train[col].unique())
    ordinal_maps[col] = unique_vals
    
    # Map values for both train and test
    train[col] = train[col].map({v: i for i, v in enumerate(unique_vals)}).astype(int)
    test[col] = test[col].fillna('missing').map({v: i for i, v in enumerate(unique_vals)}).astype(int)

# Prepare continuous features
continuous_features = numerical_cols + binary_cols + ordinal_cols + cyclical_transformed
scaler = StandardScaler()
train_continuous = scaler.fit_transform(train[continuous_features])
test_continuous = scaler.transform(test[continuous_features])

# Process nominal features
nominal_data = {}
label_encoders = {}
for col in nominal_cols:
    le = LabelEncoder()
    train[col] = le.fit_transform(train[col].fillna('missing').astype(str))
    test[col] = le.transform(test[col].fillna('missing').astype(str))
    label_encoders[col] = le

# Neural network architecture
continuous_input = Input(shape=(train_continuous.shape[1],), name='continuous')
nominal_inputs = []
embedding_layers = []

for col in nominal_cols:
    input_layer = Input(shape=(1,), name=f'nom_{col}')
    vocab_size = len(label_encoders[col].classes_) + 1
    embedding = Embedding(vocab_size, 10)(input_layer)
    embedding = Flatten()(embedding)
    nominal_inputs.append(input_layer)
    embedding_layers.append(embedding)

concatenated = concatenate([continuous_input] + embedding_layers)
dense = Dense(128, activation='relu')(concatenated)
dense = Dropout(0.2)(dense)
dense = Dense(64, activation='relu')(dense)
output = Dense(1, activation='sigmoid')(dense)

model = Model(inputs=[continuous_input] + nominal_inputs, outputs=output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Prepare feed data
train_nominal = [train[col].values.reshape(-1,1) for col in nominal_cols]
test_nominal = [test[col].values.reshape(-1,1) for col in nominal_cols]

early_stop = EarlyStopping(patience=3, restore_best_weights=True)

# Train model with timing
start_time = time.time()
history = model.fit([train_continuous] + train_nominal, y_train,
          epochs=20, batch_size=1024, validation_split=0.1,
          callbacks=[early_stop], verbose=1)
training_time = time.time() - start_time

# Save results
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history['accuracy'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_accuracy'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch]
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions
test_preds = model.predict([test_continuous] + test_nominal).flatten()
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 235, in _encode
    return _map_to_integer(values, uniques)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 174, in _map_to_integer
    return xp.asarray([table[v] for v in values], device=device(values))
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 174, in <listcomp>
    return xp.asarray([table[v] for v in values], device=device(values))
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 167, in __missing__
    raise KeyError(key)
KeyError: 'a885aacec'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/test/cat-in-the-dat-ii/cat-in-the-dat-ii_fixed.py", line 74, in <module>
    test[col] = le.transform(test[col].fillna('missing').astype(str))
  File "/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_label.py", line 134, in transform
    return _encode(y, uniques=self.classes_)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 237, in _encode
    raise ValueError(f"y contains previously unseen labels: {e}")
ValueError: y contains previously unseen labels: 'a885aacec'
</Error>