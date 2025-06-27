HYPERPARAMETER_BANK = {
  "synthetic_mlp": {
    "tags": ["classification","multiclass","tabular","low_features"],
    "description": "Toy MLP on 2–5 features",
    "params": {
      "layers":        {"type":"int",   "min":1,   "max":4,  "step":1},
      "units":         {"type":"int",   "min":50,  "max":100,"step":25},
      "activation":    {"type":"choice","values":["relu","tanh"]},
      "dropout":       {"type":"float", "min":0.0,  "max":0.3, "step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop"]},
      "learning_rate": {"type":"float", "min":1e-4,  "max":1e-3,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[16,32,64,128,256,512]},
      "epochs":        {"type":"int",   "min":10,   "max":35, "step":5},
    },
    "advice": [
      "Automatic tuning finds strong baselines on clean synthetic data, but watch for overfitting as you scale to real data.",
      "If performance plateaus early, try adding dropout up to 0.3 or reducing learning rate by an order of magnitude.",
      "For very small feature counts (<5), smaller batch sizes (16–32) often generalize better."
    ],
    "source": "Shawki et al. SPMB 2021"
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

  "image_convnet": {
    "tags": ["classification","image","binary","multiclass","medium_features"],
    "description": "ConvNet on 512×512 patches",
    "params": {
      "filters":       {"type":"int","min":16,"max":64,"step":16},
      "kernel_size":   {"type":"choice","values":[3,5]},
      "pool_size":     {"type":"choice","values":[2,3]},
      "dense_units":   {"type":"int","min":64,"max":128,"step":32},
      "dropout":       {"type":"float","min":0.25,"max":0.5,"step":0.05},
      "optimizer":     {"type":"choice","values":["adam","sgd"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-3,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[8,16,32]},
      "epochs":        {"type":"int","min":10,"max":35,"step":5},
    },
    "advice": [
      "Automatic tuning may underperform on noisy real-world images—consider adding data augmentation outside the tuner.",
      "Dropout of 0.25–0.5 after dense layers helps, but adding BatchNormalization can sometimes yield better stability.",
      "When in doubt, fix batch_size to 32 and focus on tuning filters and learning rate first."
    ],
    "source": "Shawki et al. (cancer images)"
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
    "description": "Generic regression MLP",
    "params": {
      "layers":        {"type":"int","min":2,"max":6,"step":1},
      "units":         {"type":"int","min":64,"max":512,"step":64},
      "activation":    {"type":"choice","values":["relu","tanh","selu"]},
      "dropout":       {"type":"float","min":0.0,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam","rmsprop","sgd"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[64,128,256,512]},
      "epochs":        {"type":"int","min":10,"max":50,"step":5},
      "output_activation": {"type":"choice","values":["linear"]},
      "loss":              {"type":"choice","values":["mse","mae"]}
    },
    "advice": [
      "For regression, monitor both MAE and MSE; sometimes a slightly higher MSE but lower MAE indicates better robustness to outliers.",
      "Scale your targets (e.g. StandardScaler) before training if they have large variance—this can stabilize learning rates.",
      "If you see vanishing gradients, switch activation from ReLU to SELU with appropriate AlphaDropout."
    ],
    "source": "Adapted from Shawki et al. (classification scaffold)"
  }

}


