<Code>
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

# Apply preprocessing and convert to dense arrays
X_train = preprocessor.fit_transform(train_df).toarray()
X_test = preprocessor.transform(test_df).toarray()

# ========== Keras-Tuner Integration Start ==========
import keras_tuner as kt
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

n_features = X_train.shape[1]
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', 2, 6)
        units = hp.Int('units', 64, 512, step=64)
        act = hp.Choice('activation', ['relu', 'tanh', 'selu'])
        drop = hp.Float('dropout', 0.0, 0.5)
        opt = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
        lr = hp.Float('learning_rate', 0.0001, 0.01, sampling='log')

        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation=act)(x)
            x = Dropout(drop)(x)
        x = Dense(1, activation='sigmoid')(x)
        model = Model(inputs, x)
        model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def fit(self, hp, model, *args, **kwargs):
        # Dynamically adjust batch size based on available training samples
        x_data = args[0]
        validation_split = kwargs.get('validation_split', 0.0)
        num_train_samples = int(x_data.shape[0] * (1 - validation_split))
        
        possible_batch_sizes = [64, 128, 256, 512]
        possible_batch_sizes = [bs for bs in possible_batch_sizes if bs <= num_train_samples]
        if not possible_batch_sizes:
            possible_batch_sizes = [num_train_samples]
            
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', possible_batch_sizes),
            epochs=hp.Int('epochs', 10, 50, step=5),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    project_name='depression_tuner'
)

tuner.search(
    X_train, y_train,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters()[0]
model = tuner.hypermodel.build(best_hps)
# =========== Keras-Tuner Integration End ===========

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
</Code>

<Error>
Traceback (most recent call last):
  File "/app/test/playground-series-s4e11/playground-series-s4e11_fixed.py", line 52, in <module>
    X_train = preprocessor.fit_transform(train_df).toarray()
AttributeError: 'numpy.ndarray' object has no attribute 'toarray'
</Error>