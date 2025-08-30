<Code>
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load and prepare data
train = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/train.csv.zip')
test = pd.read_csv('test/tabular-playground-series-may-2022/tabular-playground-series-may-2022/test.csv.zip')

# Separate features and target
X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']
X_test = test.drop(['id'], axis=1)
test_ids = test['id']

# Process categorical feature
le = LabelEncoder()
X_train['f_27'] = le.fit_transform(X_train['f_27'])
X_test['f_27'] = le.transform(X_test['f_27'])

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split validation set
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Build model
model = Sequential([
    Dense(256, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy',
              metrics=['AUC'])

# Train with early stopping
es = EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=1024,
                    callbacks=[es],
                    verbose=2)

# Generate predictions
test_preds = model.predict(X_test).flatten()

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'target': test_preds})
submission.to_csv('submission.csv', index=False)
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/tabular-playground-series-may-2022/tabular-playground-series-may-2022_solution.py", line 24, in <module>
    X_test['f_27'] = le.transform(X_test['f_27'])
  File "/usr/local/lib/python3.10/dist-packages/sklearn/preprocessing/_label.py", line 134, in transform
    return _encode(y, uniques=self.classes_)
  File "/usr/local/lib/python3.10/dist-packages/sklearn/utils/_encode.py", line 237, in _encode
    raise ValueError(f"y contains previously unseen labels: {e}")
ValueError: y contains previously unseen labels: 'BAAABADLAC'
</Error>