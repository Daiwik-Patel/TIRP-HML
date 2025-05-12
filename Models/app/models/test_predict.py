import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Models.app.models.model import Autoencoder # Correct import based on your project structure

# Load the trained Autoencoder model
autoencoder = Autoencoder(input_dim=42)  # Make sure to adjust input_dim based on your model's input size
autoencoder.load_state_dict(torch.load(r"./autoencoder.pth"))
autoencoder.eval()

# Load and preprocess the sample data (ensure correct relative path)
sample_file_path = r""  # Relative path to your data file
df = pd.read_csv(sample_file_path)

# Assume the dataset is in the same format and preprocess accordingly
# Convert categorical features to numerical (if applicable)
df = pd.get_dummies(df)

# Separate features and labels (assuming label is in the last column, adjust as necessary)
features = df.iloc[:, :-1].values  # All columns except the last one
labels = df.iloc[:, -1].values  # Last column (assuming it's the label)

# Scaling the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Get predictions from the Autoencoder
with torch.no_grad():  # No need to compute gradients for inference
    input_tensor = torch.tensor(features_scaled, dtype=torch.float32)
    reconstructions = autoencoder(input_tensor).numpy()  # Get the reconstructed output
    loss = ((input_tensor.numpy() - reconstructions) ** 2).mean(axis=1)  # Compute reconstruction loss (MSE)

# Example: Classify based on reconstruction error threshold (adjust threshold as needed)
threshold = 0.5  # Example threshold, adjust based on your model's characteristics
predictions = ['Normal' if l < threshold else 'Anomaly' for l in loss]

# Print the predictions
print(f"Predictions: {predictions}")
