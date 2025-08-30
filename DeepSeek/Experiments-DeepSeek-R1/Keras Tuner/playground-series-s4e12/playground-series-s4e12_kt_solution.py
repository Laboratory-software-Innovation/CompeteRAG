
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
train = pd.read_csv('/content/train.csv')
test = pd.read_csv('/content/test.csv')

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

# --- REPLACED MODEL DEFINITION BLOCK WITH KERAS-TUNER CODE ---
import keras_tuner as kt
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential

n_features = X_processed.shape[1]

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        model = Sequential()

        # Hyperparameters from bank
        layers = hp.Int('layers', 2, 8, step=1)
        units = hp.Int('units', 64, 1024, step=64)
        dropout = hp.Float('dropout', 0.0, 0.5, step=0.1)
        lr = hp.Float('learning_rate', 1e-5, 0.01, sampling='log')
        batch_size = hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024])

        # Base architecture
        model.add(Dense(units, activation='relu', input_shape=(n_features,)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))

        # Additional layers
        for _ in range(layers-1):
            model.add(Dense(units, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(1, activation='linear'))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='mean_squared_logarithmic_error',
            metrics=[tf.keras.metrics.RootMeanSquaredError(name='rmse')])
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.get('batch_size'),
            epochs=hp.Int('epochs', 20, 200, step=10),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='insurance_tuner'
)

tuner.search(
    X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
               ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)]
)

best_hp = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hp)

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
    'training_rmse': history.history['rmse'][-1],
    'training_loss': history.history['loss'][-1],
    'validation_rmse': history.history['val_rmse'][-1],
    'validation_loss': history.history['val_loss'][-1],
    'training_time': training_time  # Added training time here
}

    # Save the dictionary to a JSON file
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    print("Training results saved to results.json")