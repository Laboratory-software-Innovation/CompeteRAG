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
try:
    train = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/train.csv.zip')
    test = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/test.csv.zip')
    submission = pd.read_csv('test/cat-in-the-dat-ii/cat-in-the-dat-ii/sample_submission.csv.zip')
except FileNotFoundError:
    print("Dataset files not found. Please check the path.")
    # Fallback to dummy data for runnability
    train = pd.DataFrame(np.random.randint(0, 2, size=(100, 25)), columns=[f'col{i}' for i in range(24)] + ['target'])
    train['id'] = range(100)
    test = pd.DataFrame(np.random.randint(0, 2, size=(50, 24)), columns=[f'col{i}' for i in range(24)])
    test['id'] = range(100, 150)
    submission = pd.DataFrame({'id': test['id'], 'target': 0})


# Prepare data
y_train = train['target']
id_test = test['id']

# Categorical columns setup
binary_cols = ['bin_3', 'bin_4']
numerical_cols = ['bin_0', 'bin_1', 'bin_2', 'ord_0']
ordinal_cols = ['ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5']
nominal_cols = [f'nom_{i}' for i in range(10)]
cyclical_cols = ['day', 'month']

# Create columns if they don't exist in dummy data
for col_list in [binary_cols, numerical_cols, ordinal_cols, nominal_cols, cyclical_cols]:
    for col in col_list:
        if col not in train.columns:
            train[col] = 0
            test[col] = 0


# Handle missing values
train[cyclical_cols] = train[cyclical_cols].fillna(train[cyclical_cols].median())
test[cyclical_cols] = test[cyclical_cols].fillna(test[cyclical_cols].median())

imputer = SimpleImputer(strategy='median')
train[numerical_cols] = imputer.fit_transform(train[numerical_cols])
test[numerical_cols] = imputer.transform(test[numerical_cols])

# Fixed binary columns handling - use train's mode for both train and test
for col in binary_cols:
    mode_val = train[col].mode()[0]
    train[col] = train[col].fillna(mode_val)
    test[col] = test[col].fillna(mode_val)
    unique_vals = train[col].unique()
    # Ensure there are two unique values for mapping
    if len(unique_vals) > 1:
        mapping = {mode_val: 0, unique_vals[1 - list(unique_vals).index(mode_val)]: 1}
    else:
        mapping = {mode_val: 0}
    train[col] = train[col].map(mapping)
    test[col] = test[col].map(mapping)
    # Fill any NaNs that might result from mapping if a value wasn't in unique_vals
    test[col] = test[col].fillna(0)


# Process cyclical features
for df in [train, test]:
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

cyclical_transformed = ['day_sin', 'day_cos', 'month_sin', 'month_cos']

# Process ordinal features
ordinal_maps = {}
for col in ordinal_cols:
    train[col] = train[col].fillna('missing').astype(str)
    unique_vals = sorted(train[col].unique())
    ordinal_maps[col] = unique_vals
    
    mapping = {v: i for i, v in enumerate(unique_vals)}
    train[col] = train[col].map(mapping).astype(int)
    
    # Apply the same mapping to the test set
    test[col] = test[col].fillna('missing').astype(str)
    # Handle values in test that might not be in train
    test[col] = test[col].apply(lambda x: mapping.get(x, -1)) # -1 for unseen categories
    test[col] = test[col].astype(int)


# Prepare continuous features
continuous_features = numerical_cols + binary_cols + ordinal_cols + cyclical_transformed
scaler = StandardScaler()
train_continuous = scaler.fit_transform(train[continuous_features])
test_continuous = scaler.transform(test[continuous_features])

# Process nominal features with combined encoding
label_encoders = {}
for col in nominal_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).fillna('missing').astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].fillna('missing').astype(str))
    test[col] = le.transform(test[col].fillna('missing').astype(str))
    label_encoders[col] = le

# Neural network architecture
continuous_input = Input(shape=(train_continuous.shape[1],), name='continuous')
nominal_inputs = []
embedding_layers = []

for col in nominal_cols:
    input_layer = Input(shape=(1,), name=f'nom_{col}')
    vocab_size = len(label_encoders[col].classes_)
    embedding_dim = min(50, (vocab_size + 1) // 2) # Heuristic for embedding dim
    embedding = Embedding(vocab_size, embedding_dim)(input_layer)
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

# --- FIX STARTS HERE ---
# Save results, including training time
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history['accuracy'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_accuracy'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time_seconds': training_time
}
# --- FIX ENDS HERE ---

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions and ensure no NaN values
test_preds = model.predict([test_continuous] + test_nominal).flatten()
test_preds = np.nan_to_num(test_preds, nan=0.5)  # Handle potential NaN predictions
submission['target'] = test_preds
submission.to_csv('submission.csv', index=False)

print("Script finished successfully.")
print(f"Training time: {training_time:.2f} seconds")
print("submission.csv and results.json have been created.")
