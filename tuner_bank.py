HYPERPARAMETER_BANK = {
  "synthetic_mlp": {
    "tags": ["classification", "multiclass", "tabular", "low_features", "multi-label"],
    "description": "Scaled MLP on 2–5 features for larger datasets",
    "params": {
      "layers":        {"type":"int","min":2,"max":8,"step":1},
      "units":         {"type":"int","min":64,"max":1024,"step":64},
      "activation":    {"type":"choice","values":["relu"]},
      "dropout":       {"type":"float","min":0.0,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam"]},
      "learning_rate": {"type":"float","min":1e-5,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[32,64,128,256,512,1024]},
      "epochs":        {"type":"int","min":20,"max":200,"step":10}
    },
  },


  "tabular_classification": {
    "tags": ["classification", "multiclass", "tabular", "low_features", "multi-label"],
    "description": "Binary classification on ~50–500 features",
    "params": {
      "layers":        {"type":"int","min":2,"max":8,"step":1},
      "units":         {"type":"int","min":64,"max":512,"step":64},
      "activation":    {"type":"choice","values":["relu"]},
      "dropout":       {"type":"float","min":0.0,"max":0.5,"step":0.1},
      "optimizer":     {"type":"choice","values":["adam"]},
      "learning_rate": {"type":"float","min":1e-4,"max":1e-2,"sampling":"log"},
      "batch_size":    {"type":"choice","values":[64,128,256,512]},
      "epochs":        {"type":"int","min":10,"max":50,"step":5},
    },
  },
  
  "ts_lstm_forecasting": {
  "tags":       ["time-series", "forecasting", "sequence"],
  "description":"Univariate/multivariate forecasting with stacked LSTM layers",
  "params": {
    "layers":        {"type":"int",    "min":1,   "max":4,   "step":1},
    "units":         {"type":"int",    "min":32,  "max":256, "step":32},    
    "dense_units":   {"type":"int",    "min":16,  "max":128, "step":16},   
    "activation":    {"type":"choice", "values":["tanh"]},                  
    "dropout":       {"type":"float",  "min":0.0,  "max":0.5,  "step":0.1},
    "optimizer":     {"type":"choice", "values":["adam"]},
    "learning_rate": {"type":"float",  "min":1e-5, "max":1e-2,"sampling":"log"},
    "batch_size":    {"type":"choice", "values":[32,64,128,256]},
    "epochs":        {"type":"int",    "min":10,  "max":100, "step":10}
  },
},


  "regression_mlp": {
    "tags": ["regression", "tabular", "low_features", "medium_features", "missing-values"],
    "description": "Scaled MLP for regression on low-dimensional tabular data",
    "params": {
      "layers":            {"type":"int",   "min":2,    "max":8,   "step":1},
      "units":             {"type":"int",   "min":64,   "max":1024,"step":64},
      "activation":        {"type":"choice","values":["relu"]},
      "dropout":           {"type":"float", "min":0.0,   "max":0.5,  "step":0.1},
      "optimizer":         {"type":"choice","values":["adam"]},
      "learning_rate":     {"type":"float", "min":1e-5,  "max":1e-2,"sampling":"log"},
      "batch_size":        {"type":"choice","values":[32,64,128,256,512,1024]},
      "epochs":            {"type":"int",   "min":20,   "max":200, "step":10},
    },
  }

}
