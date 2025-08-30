import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

# Load data
train = pd.read_csv('test/playground-series-s4e12/playground-series-s4e12/train.csv')
test = pd.read_csv('test/playground-series-s4e12/playground-series-s4e12/test.csv')

# Remove target outliers
y = train['Premium Amount']
q1, q3 = y.quantile(0.25), y.quantile(0.75)
iqr = q3 - q1
train = train[(y >= q1 - 1.5*iqr) & (y <= q3 + 1.5*iqr)].copy()
y = train['Premium Amount']

# Process dates
def process_dates(df):
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Policy_Year'] = df['Policy Start Date'].dt.year
    df['Policy_Month'] = df['Policy Start Date'].dt.month
    return df.drop('Policy Start Date', axis=1)

train = process_dates(train)
test = process_dates(test)

# Define features
X = train.drop(['id', 'Premium Amount'], axis=1)
test_ids = test['id'] # Save test ids before dropping
X_test = test.drop('id', axis=1)


numerical_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
                      'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
                      'Policy_Year', 'Policy_Month']
categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation',
                        'Location', 'Policy Type', 'Customer Feedback', 'Smoking Status',
                        'Exercise Frequency', 'Property Type']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numerical_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ]), categorical_features)
])

X_processed = preprocessor.fit_transform(X)
X_test_processed = preprocessor.transform(X_test)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build model
model = Sequential([
    Dense(512, activation='relu', input_shape=(X_processed.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='mean_squared_logarithmic_error',
              metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])

# Define callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True)

# Train model with timing
start_time = time.time()
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=256,
    callbacks=[early_stop, checkpoint],
    verbose=1
)
training_time = time.time() - start_time
print(f"\nTraining completed in {training_time:.2f} seconds.")

# Generate predictions
model.load_weights('best_model.h5')
test_preds = model.predict(X_test_processed).flatten()
test_preds = np.clip(test_preds, a_min=0, a_max=None)

# Create submission
submission = pd.DataFrame({
    'id': test_ids,
    'Premium Amount': test_preds
})
submission.to_csv('submission.csv', index=False)
print("Submission file created successfully.")


# --- ADDED BLOCK: Save training results ---
if history and history.history:
    # Get the number of epochs the model actually ran for
    final_epoch = len(history.history['loss']) - 1
    
    # Create a dictionary with the final results
    results = {
        'final_training_loss': history.history['loss'][final_epoch],
        'final_training_rmse': history.history['rmse'][final_epoch],
        'final_validation_loss': history.history['val_loss'][final_epoch],
        'final_validation_rmse': history.history['val_rmse'][final_epoch],
        'training_time_seconds': training_time
    }

    # Save the dictionary to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("Training results saved to results.json")
