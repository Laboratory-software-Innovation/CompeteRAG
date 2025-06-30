HYPERPARAMETER_BANK = {
  "synthetic_mlp": {
    "tags": ["classification","multiclass","tabular","low_features"],
    "description": "Scaled MLP on 2–5 features for larger datasets",
    "params": {
      "layers":        {"type":"int","min":2,"max":8,"step":1},
      "units":         {"type":"int","min":64,"max":1024,"step":64},
      "activation":    {"type":"choice","values":["relu","tanh","selu"]},
      "dropout":       {"type":"float","min":0.0,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop","sgd"]},
      "learning_rate": {"type":"float","min":1e-5,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[32,64,128,256,512,1024]},
      "epochs":        {"type":"int","min":20,"max":200,"step":10}
    },
    "advice": [
      "For low-dimensional data but large samples, start with moderate batch sizes (128–512) and tune up if GPU memory allows.",
      "Deeper nets (4–8 layers) can capture complex feature interactions—use SELU with AlphaDropout if you go deep.",
      "Use learning-rate search on a log scale from 1e-5 to 1e-2 to find stability vs. convergence trade-offs."
    ],
    "source": "Adapted from the Keras Tuner Guide (https://keras.io/keras_tuner/)"
  },


  "tabular_classification": {
    "tags": ["classification","binary","tabular","medium_features"],
    "description": "Binary classification on ~50–500 features",
    "params": {
      "layers":        {"type":"int","min":2,"max":6,"step":1},
      "units":         {"type":"int","min":64,"max":512,"step":64},
      "activation":    {"type":"choice","values":["relu","tanh","selu"]},
      "dropout":       {"type":"float","min":0.0,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop","sgd"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[64,128,256,512]},
      "epochs":        {"type":"int","min":10,"max":50,"step":5},
    },
    "advice": [
      "Larger networks (up to 6 layers) capture complex feature interactions but require stronger regularization (dropout ≥0.3).",
      "Use log-scale learning-rate search from 1e-4 to 1e-2 to find the sweet spot between convergence speed and stability.",
      "Test both adaptive (Adam/RMSprop) and plain SGD—sometimes SGD with a small LR gives better generalization on tabular data."
    ],
    "source": "Combined from Shawki et al. & Dudko et al."
  },

  "text_cnn": {
    "tags": ["classification","text","multiclass","medium_features","missing-values"],
    "description": "NLP text classification on retrospectives",
    "params": {
      "conv_blocks":   {"type":"int","min":1,"max":5,"step":1},
      "filters":       {"type":"int","min":16,"max":256,"step":16},
      "kernel_size":   {"type":"choice","values":[3,5]},
      "activation":    {"type":"choice","values":["relu","tanh"]},
      "dropout":       {"type":"float","min":0.1,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop","sgd"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[32,64,128]},
      "epochs":        {"type":"int","min":10,"max":50,"step":5},
      "imputation":    {"type":"choice","values":["mean","median","knn"]},
      "add_indicator": {"type":"boolean"}
    },
    "advice": [
      "Include missing-value imputation as part of the search (mean/median/KNN) when your text features have gaps.",
      "Higher dropout (up to 0.5) helps prevent overfitting on small text corpora, but monitor validation loss closely.",
      "For text CNNs, kernel sizes of 3 usually capture sufficient n-gram patterns—only try 5 if you suspect longer dependencies."
    ],
    "source": "Rogachev & Melikhova IOP EES 2020"
  },

  "ts_lstm_cnn": {
    "tags": ["classification","time-series","binary","medium_features"],
    "description": "EEG seizure vs. background detection",
    "params": {
      "conv_filters":  {"type":"int","min":16,"max":32,"step":16},
      "kernel_size":   {"type":"choice","values":[3]},
      "lstm_units":    {"type":"int","min":64,"max":256,"step":64},
      "dense_units":   {"type":"int","min":32,"max":128,"step":32},
      "dropout":       {"type":"float","min":0.1,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-3,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[32,64,128]},
      "epochs":        {"type":"int","min":10,"max":35,"step":5},
    },
    "advice": [
      "Time-series models can overfit quickly—use lower dropout rates (0.1–0.3) and monitor sequence-level validation metrics.",
      "Tune LSTM units carefully: too many (>128) may slow convergence without accuracy gains.",
      "Prefer Adam for sequence data, but try RMSprop if you see unstable training loss."
    ],
    "source": "Shawki et al. (EEG signals)"
  },

  "regression_mlp": {
    "tags": ["regression","tabular","low_features","medium_features"],
    "description": "Scaled MLP for regression on low-dimensional tabular data",
    "params": {
      "layers":            {"type":"int",   "min":2,    "max":8,   "step":1},
      "units":             {"type":"int",   "min":64,   "max":1024,"step":64},
      "activation":        {"type":"choice","values":["relu","tanh","selu"]},
      "dropout":           {"type":"float", "min":0.0,   "max":0.5,  "step":0.1},
      "optimizer":         {"type":"choice","values":["adam","rmsprop","sgd"]},
      "learning_rate":     {"type":"float", "min":1e-5,  "max":1e-2,"sampling":"log"},
      "batch_size":        {"type":"choice","values":[32,64,128,256,512,1024]},
      "epochs":            {"type":"int",   "min":20,   "max":200, "step":10},
      "output_activation": {"type":"choice","values":["linear"]},
      "loss":              {"type":"choice","values":["mse","mae","huber"]}
    },
    "advice": [
      "Use a linear output activation and MSE or Huber loss for smooth regression targets.",
      "For deeper networks (≥6 layers), SELU + AlphaDropout can help maintain self-normalizing activations.",
      "Tune learning rate on a log scale (1e-5 to 1e-2) and use moderate batch sizes (128–512) for stable convergence."
    ],
    "source": "Adapted from the Keras Tuner Guide (https://keras.io/keras_tuner/)"
  }

}


