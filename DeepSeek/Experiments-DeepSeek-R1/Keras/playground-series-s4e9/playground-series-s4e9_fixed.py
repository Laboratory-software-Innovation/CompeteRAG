import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Concatenate, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
import time

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

# Prepare model inputs
cat_inputs = []
for col in cat_cols:
    input_layer = Input(shape=(1,), name=f'input_{col}')
    cat_inputs.append(input_layer)

# Create embeddings
cat_embeddings = []
for i, col in enumerate(cat_cols):
    vocab_size = len(label_encoders[col].classes_)
    # A common rule of thumb for embedding size
    embedding_size = min(50, (vocab_size + 1) // 2)
    emb = Embedding(vocab_size, embedding_size)(cat_inputs[i])
    cat_embeddings.append(Flatten()(emb))

# Numerical stream
numerical_input = Input(shape=(len(numerical_cols),))
numerical_processed = Dense(64, activation='relu')(numerical_input)

# Concatenate and build model
concat = Concatenate()(cat_embeddings + [numerical_processed])
x = Dense(512, activation='relu')(concat)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)
output = Dense(1)(x)

model = Model(inputs=cat_inputs + [numerical_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='mse', metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Prepare input data
train_cat = [X_train[col].values for col in cat_cols]
train_num = X_train[numerical_cols].values
val_cat = [X_val[col].values for col in cat_cols]
val_num = X_val[numerical_cols].values

# Train model and time the process
start_time = time.time()
history = model.fit(
    train_cat + [train_num], y_train,
    validation_data=(val_cat + [val_num], y_val),
    epochs=100,
    batch_size=1024,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
    verbose=2
)
training_duration = time.time() - start_time

# FIXED: Save results from the best epoch, not the last one.
# Find the epoch with the best validation loss.
best_epoch = np.argmin(history.history['val_loss'])
results = {
    "training_rmse": history.history['rmse'][best_epoch],
    "training_loss": history.history['loss'][best_epoch],
    "validation_rmse": history.history['val_rmse'][best_epoch],
    "validation_loss": history.history['val_loss'][best_epoch],
    "training_time_seconds": training_duration
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Process test data
test = pd.read_csv('test/playground-series-s5e2/playground-series-s5e2/test.csv')
test_ids = test[id_col]

# Preprocess categorical test data
test_cat_data = test[cat_cols].copy()
for col in cat_cols:
    test_cat_data[col] = test_cat_data[col].fillna('Missing')
    # Use the learned classes from the training data to handle unseen labels
    valid_labels = set(label_encoders[col].classes_)
    # Any label in the test set not in `valid_labels` will be treated as 'Missing'
    test_cat_data[col] = test_cat_data[col].apply(lambda x: x if x in valid_labels else 'Missing')
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
