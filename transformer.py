import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils import plot_prediction_vs_actual, plot_losses

# Load data
traning_loss = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

filepath = 'london_weather.csv'
data = pd.read_csv(filepath)

# Convert date to datetime format
data['date'] = pd.to_datetime(data['date'])

# Define features (excluding 'snow_depth')
features = ['cloud_cover', 'sunshine', 'global_radiation', 'max_temp', 'mean_temp', 'min_temp', 'pressure']
data = data[features]
data.fillna({
    "cloud_cover" : data["cloud_cover"].mode(),
    "global_radiation" : data["global_radiation"].mean(),
    "mean_temp": data["mean_temp"].mean()
}, inplace=True)
#Drop the remaining rows that have missing values
data.dropna(inplace=True)

# Normalize features
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
sequence_length = 30  # Number of days to consider for each sequence
X, y = [], []
for i in range(len(data_scaled) - sequence_length):
    X.append(data_scaled[i:i + sequence_length])
    y.append(data_scaled[i + sequence_length])
X, y = np.array(X), np.array(y)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device=device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device=device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device=device)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device=device)

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_length, num_heads=9, num_layers=2, hidden_dim=72):
        super(TransformerRegressor, self).__init__()
        self.input_dim = input_dim
        self.seq_length = seq_length

        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)  # Apply linear transformation to input
        x = x.permute(1, 0, 2)  # PyTorch Transformer expects (seq_len, batch, hidden_dim)
        x = self.transformer_encoder(x)
        x = x[-1, :, :]  # Take the output of the last time step
        x = self.fc(x)
        return x


def train_transform():
    # Define and train the model
    model = TransformerRegressor(input_dim=len(features), output_dim=len(features), seq_length=sequence_length, num_heads=10, hidden_dim=80).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 25
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        traning_loss.append(loss.item())
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).cpu().numpy()
        actual = y_test_tensor.cpu().numpy()

    # Reverse scaling
    predictions_inv = scaler.inverse_transform(predictions)
    actual_inv = scaler.inverse_transform(actual)

    # Plot predictions vs actual values
    torch.save(model.state_dict(), 'weather_transformer_net.pth')

    plot_prediction_vs_actual(predictions_inv, actual_inv, features, save_dir='transformer_plots')
    plot_losses(loss_val=[], loss_train=traning_loss, save_dir='transformer_plots')


train_transform()
print("done training")