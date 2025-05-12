import torch
import torch.nn as nn
import joblib
import numpy as np
from Models.app.models.model import Autoencoder  # adjust if class is elsewhere

# Load models
autoencoder = Autoencoder(input_dim=42)  # use your actual input size
autoencoder.load_state_dict(torch.load("Models/app/models/autoencoder.pth"))
autoencoder.eval()

rf_model = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

def predict(input_data):
    # Step 1: Preprocess
    input_scaled = scaler.transform(input_data)

    # Step 2: Encode with autoencoder
    with torch.no_grad():
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        encoded = autoencoder.encoder(input_tensor).numpy()

    # Step 3: Predict with Random Forest
    prediction = rf_model.predict(encoded)
    return prediction
