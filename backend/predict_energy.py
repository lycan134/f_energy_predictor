from backend.neural_network import NeuralNetwork
import torch
import pandas as pd

MODEL_PATH = "models/best_model_full.pt"
NORM_STATS_PATH = "models/normalization_stats.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# (Optional) Workaround for pickled model class mismatch:
import sys
sys.modules['__main__'].NeuralNetwork = NeuralNetwork

# Load model
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model.eval()

# Load normalization stats
stats = torch.load(NORM_STATS_PATH, map_location=device)
X_mean, X_std = stats["X_mean"], stats["X_std"]
y_mean, y_std = stats["y_mean"], stats["y_std"]
eps = 1e-8

def preprocess_input(df):
    """Normalize input data just like training"""
    X = torch.tensor(df.values, dtype=torch.float32)
    X = (X - X_mean) / (X_std + eps)
    return X.to(device)

def predict_energy(input_df):
    """
    input_df: DataFrame with same columns as training X
    returns: Predicted formation energy per atom (denormalized)
    """
    X = preprocess_input(input_df)
    with torch.no_grad():
        y_pred_norm = model(X)
        y_pred = y_pred_norm * (y_std + eps) + y_mean  # denormalize
    return y_pred.cpu().numpy().flatten()
