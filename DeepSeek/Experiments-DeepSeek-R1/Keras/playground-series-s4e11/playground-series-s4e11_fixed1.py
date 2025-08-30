import pandas as pd
import numpy as np
import json
import time
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load datasets
train_df = pd.read_csv('test/playground-series-s4e11/playground-series-s4e11/train.csv')
test_df = pd.read_csv('test/playground-series-s4e11/playground-series-s4e11/test.csv')

# Save test IDs before preprocessing
test_ids = test_df['id']

# Drop irrelevant columns
drop_cols = ['id', 'Name', 'City', 'Profession', 
             'Academic Pressure', 'CGPA', 'Study Satisfaction']
train_df = train_df.drop(columns=drop_cols)
test_df = test_df.drop(columns=drop_cols)

# Process binary categorical columns
for col in ['Have you ever had suicidal thoughts ?', 'Family History of Mental Illness']:
    train_df[col] = train_df[col].map({'Yes': 1, 'No': 0}).astype(int)
    test_df[col] = test_df[col].map({'Yes': 1, 'No': 0}).astype(int)

# Separate target variable
y_train = train_df['Depression'].map({'e':0, 'p':1}).values
train_df = train_df.drop(columns=['Depression'])

# Define preprocessing components
num_features = ['Age', 'Work Pressure', 'Job Satisfaction', 
                'Work/Study Hours', 'Financial Stress']
ordinal_features = ['Sleep Duration', 'Degree']
nominal_features = ['Gender', 'Working Professional or Student', 'Dietary Habits']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_features),
    ('ordinal', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ]), ordinal_features),
    ('nominal', OneHotEncoder(handle_unknown='ignore'), nominal_features)
], remainder='passthrough')

# Apply preprocessing
X_train = preprocessor.fit_transform(train_df)
X_test = preprocessor.transform(test_df)

# Build neural network
model = Sequential([
    Dense(1024, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(768, activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train with early stopping and time tracking
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=1024,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)],
    verbose=2
)
training_time = time.time() - start_time

# Save results
last_epoch = len(history.history['loss']) - 1
results = {
    'training_accuracy': history.history['accuracy'][last_epoch],
    'training_loss': history.history['loss'][last_epoch],
    'validation_accuracy': history.history['val_accuracy'][last_epoch],
    'validation_loss': history.history['val_loss'][last_epoch],
    'training_time_seconds': training_time
}

with open('results.json', 'w') as f:
    json.dump(results, f)

# Generate predictions
test_preds = (model.predict(X_test) > 0.5).astype(int).flatten()

# Create submission file
submission = pd.DataFrame({'id': test_ids, 'Depression': test_preds})
submission.to_csv('submission.csv', index=False)