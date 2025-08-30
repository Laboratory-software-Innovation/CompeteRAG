
import pandas as pd
import json
import time
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess training data
train_df = pd.read_csv('test/playground-series-s5e6/playground-series-s5e6/train.csv')
train_df['Fertilizer Name'] = train_df['Fertilizer Name'].str.split()

# Multi-label target encoding
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df['Fertilizer Name'])

# Feature preprocessing pipeline
numerical_features = ['Temparature', 'Humidity', 'Moisture', 'Nitrogen', 'Potassium', 'Phosphorous']
categorical_features = ['Soil Type', 'Crop Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Correctly apply the preprocessor to the feature columns
X = preprocessor.fit_transform(train_df.drop(['id', 'Fertilizer Name'], axis=1))

# Neural network architecture with Keras Tuner
import keras_tuner as kt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

n_features = X.shape[1]
original_loss = 'binary_crossentropy'
original_metrics = ['accuracy']

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

class MyHyperModel(kt.HyperModel):
    def build(self, hp):
        layers = hp.Int('layers', min_value=2, max_value=8, step=1)
        units = hp.Int('units', min_value=64, max_value=1024, step=64)
        drop = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1)
        lr = hp.Float('learning_rate', min_value=1e-05, max_value=0.01, sampling='log')
        
        inputs = Input(shape=(n_features,))
        x = inputs
        for _ in range(layers):
            x = Dense(units, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(drop)(x)
        x = Dense(y.shape[1], activation='sigmoid')(x)
        
        model = Model(inputs, x)
        model.compile(
            optimizer='adam',
            loss=original_loss,
            metrics=original_metrics
        )
        return model

    def fit(self, hp, model, *args, **kwargs):
        return model.fit(
            *args,
            batch_size=hp.Choice('batch_size', [32, 64, 128, 256, 512, 1024]),
            epochs=hp.Int('epochs', min_value=20, max_value=200, step=10),
            **kwargs
        )

tuner = kt.BayesianOptimization(
    MyHyperModel(),
    objective='val_loss',
    max_trials=10,
    executions_per_trial=1,
    seed=42,
    overwrite=True,
    project_name='bayesian_tuner'
)

tuner.search(
    X, y,
    validation_split=0.2,
    callbacks=[early_stopping, checkpoint]
)

best_hps = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hps)

# --- Original training call remains below ---
# Record the start and end time of the model training
start_time = time.time()

# Train model
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history = model.fit(
    X, y,
    epochs=50,
    batch_size=512,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

training_time = time.time() - start_time
# --- End of fix ---

# Preprocess and predict test data
test_df = pd.read_csv('test/playground-series-s5e6/playground-series-s5e6/test.csv')
X_test = preprocessor.transform(test_df.drop('id', axis=1))
probabilities = model.predict(X_test)

# Generate top-3 predictions
top3_indices = np.argsort(-probabilities, axis=1)[:, :3]
predictions = [' '.join(mlb.classes_[row]) for row in top3_indices]

# Create submission file
submission = pd.DataFrame({
    'id': test_df['id'],
    'Fertilizer Name': predictions
})
submission.to_csv('submission.csv', index=False)

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
