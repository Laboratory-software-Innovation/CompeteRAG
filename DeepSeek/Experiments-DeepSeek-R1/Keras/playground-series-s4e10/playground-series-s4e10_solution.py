<Code>
import pandas as pd
import numpy as np
from category_encoders import CatBoostEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load data
train = pd.read_csv('test/playground-series-s4e10/playground-series-s4e10/train.csv')
test = pd.read_csv('test/playground-series-s4e10/playground-series-s4e10/test.csv')

# Prepare features and target
X = train.drop(['id', 'loan_status'], axis=1)
y = train['loan_status']
X_test = test.drop('id', axis=1)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Identify categorical columns
cat_cols = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']
num_cols = [col for col in X_train.columns if col not in cat_cols]

# Encode categorical features
encoder = CatBoostEncoder()
X_train_encoded = encoder.fit_transform(X_train[cat_cols], y_train)
X_val_encoded = encoder.transform(X_val[cat_cols])
X_test_encoded = encoder.transform(X_test[cat_cols])

# Merge encoded features with numerical features
X_train = pd.concat([X_train[num_cols], X_train_encoded], axis=1)
X_val = pd.concat([X_val[num_cols], X_val_encoded], axis=1)
X_test = pd.concat([X_test[num_cols], X_test_encoded], axis=1)

# Standardize numerical features
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Build model
input_shape = X_train.shape[1]
inputs = Input(shape=(input_shape,))
X = inputs
for units in [512, 256, 128, 64]:
    X = Dense(units)(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
X = Dense(1)(X)
X = BatchNormalization()(X)
outputs = Activation('sigmoid')(X)
model = Model(inputs=inputs, outputs=outputs)

# Compile with learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=3200,
    decay_rate=0.9,
    staircase=True)
model.compile(loss='binary_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule),
              metrics=['AUC'])

# Early stopping and training
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Generate predictions
test_pred = model.predict(X_test).flatten()

# Create submission file
submission = pd.DataFrame({'id': test['id'], 'loan_status': test_pred})
submission.to_csv('submission.csv', index=False)
</Code>


<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s4e10/playground-series-s4e10_solution.py", line 4, in <module>
    from category_encoders import CatBoostEncoder
ModuleNotFoundError: No module named 'category_encoders'
</Error>
