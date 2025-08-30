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

# Load data
train = pd.read_csv('test/playground-series-s4e9/playground-series-s4e9/train.csv')
test = pd.read_csv('test/playground-series-s4e9/playground-series-s4e9/test.csv')

# Feature engineering for engine
def parse_engine(s):
    displacement = float(re.findall(r'\d+\.\d+', s)[0]) if re.findall(r'\d+\.\d+', s) else 0.0
    cylinders = int(re.findall(r'V(\d+)', s)[0]) if re.findall(r'V(\d+)', s) else 0
    return displacement, cylinders

train['displacement'], train['cylinders'] = zip(*train['engine'].apply(parse_engine))
test['displacement'], test['cylinders'] = zip(*test['engine'].apply(parse_engine))

# Handle missing values
for col in ['fuel_type', 'accident', 'clean_title']:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

# Convert binary features
train['accident'] = train['accident'].map({'Yes':1, 'No':0})
test['accident'] = test['accident'].map({'Yes':1, 'No':0})
train['clean_title'] = train['clean_title'].map({'Yes':1, 'No':0})
test['clean_title'] = test['clean_title'].map({'Yes':1, 'No':0})

# Encode categorical features and store label encoders
categorical_cols = ['brand', 'model', 'fuel_type', 'transmission', 'ext_col', 'int_col']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0)
    le.fit(combined)
    train[col] = le.transform(train[col])
    test[col] = le.transform(test[col])
    label_encoders[col] = le

# Prepare features
numerical_cols = ['model_year', 'milage', 'displacement', 'cylinders', 'accident', 'clean_title']
X_numerical = train[numerical_cols].values
X_categorical = [train[col].values for col in categorical_cols]
y = train['price'].values

# Split data with proper array unpacking
(X_num_train, X_num_val, 
 cat1_train, cat1_val,
 cat2_train, cat2_val,
 cat3_train, cat3_val,
 cat4_train, cat4_val,
 cat5_train, cat5_val,
 cat6_train, cat6_val,
 y_train, y_val) = train_test_split(
    X_numerical,
    X_categorical[0],
    X_categorical[1],
    X_categorical[2],
    X_categorical[3],
    X_categorical[4],
    X_categorical[5],
    y,
    test_size=0.2,
    random_state=42
)

X_cat_train = [cat1_train, cat2_train, cat3_train, cat4_train, cat5_train, cat6_train]
X_cat_val = [cat1_val, cat2_val, cat3_val, cat4_val, cat5_val, cat6_val]

# Scale numerical features with zero variance handling
mean = np.mean(X_num_train, axis=0)
std = np.std(X_num_train, axis=0)
std[std == 0] = 1.0  # Prevent division by zero for constant features
X_num_train = (X_num_train - mean) / std
X_num_val = (X_num_val - mean) / std

# Build model with correct embedding dimensions
numerical_input = Input(shape=(len(numerical_cols),))
categorical_inputs = []
embedding_layers = []

for col in categorical_cols:
    max_cat = len(label_encoders[col].classes_)
    embed_size = min(50, max_cat//2 + 1)
    input_layer = Input(shape=(1,))
    embedding = Embedding(input_dim=max_cat, output_dim=embed_size)(input_layer)
    embedding_layers.append(Flatten()(embedding))
    categorical_inputs.append(input_layer)

concatenated = Concatenate()([numerical_input] + embedding_layers)
dense = Dense(256, activation='relu')(concatenated)
dense = Dropout(0.3)(dense)
dense = Dense(128, activation='relu')(dense)
dense = Dropout(0.2)(dense)
output = Dense(1)(dense)

model = Model(inputs=[numerical_input] + categorical_inputs, outputs=output)
model.compile(optimizer='adam', loss='mse')

# Train model with timing
start_time = time.time()
history = model.fit(
    [X_num_train] + [x.reshape(-1,1) for x in X_cat_train],
    y_train,
    validation_data=([X_num_val] + [x.reshape(-1,1) for x in X_cat_val], y_val),
    epochs=100,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=1
)
training_time = time.time() - start_time

# Save training results
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history.get('accuracy', [None])[-1],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history.get('val_accuracy', [None])[-1],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions
test_num = (test[numerical_cols].values - mean) / std  # Apply same scaling
test_cat = [test[col].values.reshape(-1,1) for col in categorical_cols]
predictions = model.predict([test_num] + test_cat).flatten()

# Ensure no null values in predictions
assert not np.isnan(predictions).any(), "Predictions contain NaN values"

# Create submission
submission = pd.DataFrame({'id': test['id'], 'price': predictions})
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s4e9/playground-series-s4e9_fixed.py", line 139, in <module>
    assert not np.isnan(predictions).any(), "Predictions contain NaN values"
AssertionError: Predictions contain NaN values
# Handle missing values
for col in ['fuel_type', 'accident', 'clean_title']:
    train[col] = train[col].fillna(train[col].mode()[0])
    test[col] = test[col].fillna(test[col].mode()[0])

# FIX: Add imputation for numerical columns to prevent NaNs
numerical_cols_to_impute = ['model_year', 'milage', 'displacement', 'cylinders']
for col in numerical_cols_to_impute:
    median_val = train[col].median() # Use median from training data
    train[col].fillna(median_val, inplace=True)
    test[col].fillna(median_val, inplace=True)

Learning Rate is Too High: This is the #1 cause. If the learning rate is too aggressive, the updates to the model's weights are too large, causing them to fly off towards infinity.
# FIX: Lower the learning rate significantly to ensure stable training
from tensorflow.keras.optimizers import Adam
model.compile(optimizer=Adam(learning_rate=1e-5), loss='mse')

Input Data is Not Scaled: Neural networks are very sensitive to the scale of the input data. If you have some features with very large values (e.g., house prices in the millions) and others with small values (e.g., number of rooms from 1-5), the training process can become unstable.
# FIX: Use StandardScaler for robust scaling after splitting the data
scaler = StandardScaler()
X_num_train = scaler.fit_transform(X_num_train)
X_num_val = scaler.transform(X_num_val)
</Error>