<Code>
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

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

# Process ordinal features
ordinal_maps = {
    'ord_1': sorted(train['ord_1'].dropna().unique()),
    'ord_2': sorted(train['ord_2'].dropna().unique()),
    'ord_3': sorted(train['ord_3'].dropna().unique()),
    'ord_4': sorted(train['ord_4'].dropna().unique()),
    'ord_5': sorted(train['ord_5'].dropna().unique())
}

for col in ordinal_cols:
    train[col] = train[col].fillna('missing').map({v: i for i, v in enumerate(ordinal_maps[col])}).astype(int)
    test[col] = test[col].fillna('missing').map({v: i for i, v in enumerate(ordinal_maps[col])}).astype(int)

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

# Prepare feed data
train_nominal = [train[col].values.reshape(-1,1) for col in nominal_cols]
test_nominal = [test[col].values.reshape(-1,1) for col in nominal_cols]

early_stop = EarlyStopping(patience=3, restore_best_weights=True)
model.fit([train_continuous] + train_nominal, y_train, 
          epochs=20, batch_size=1024, validation_split=0.1, callbacks=[early_stop])

# Generate predictions
test_preds = model.predict([test_continuous] + test_nominal).flatten()
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/cat-in-the-dat-ii/cat-in-the-dat-ii_solution.py", line 59, in <module>
    train[col] = train[col].fillna('missing').map({v: i for i, v in enumerate(ordinal_maps[col])}).astype(int)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/generic.py", line 6662, in astype
    new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/internals/managers.py", line 430, in astype
    return self.apply(
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/internals/managers.py", line 363, in apply
    applied = getattr(b, f)(**kwargs)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/internals/blocks.py", line 784, in astype
    new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/astype.py", line 237, in astype_array_safe
    new_values = astype_array(values, dtype, copy=copy)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/astype.py", line 182, in astype_array
    values = _astype_nansafe(values, dtype, copy=copy)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/astype.py", line 101, in _astype_nansafe
    return _astype_float_to_int_nansafe(arr, dtype, copy)
  File "/usr/local/lib/python3.10/dist-packages/pandas/core/dtypes/astype.py", line 145, in _astype_float_to_int_nansafe
    raise IntCastingNaNError(
pandas.errors.IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
</Error>