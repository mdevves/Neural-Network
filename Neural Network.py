import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "data_for_NN.csv"
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Dataset not found at {file_path}")

df = pd.read_csv(file_path)

# Extract features (5 inputs) and performance metrics (8 outputs)
scaler_X =  StandardScaler()
scaler_y =  StandardScaler()

X = scaler_X.fit_transform(df.iloc[:, :5].values)  # Normalize inputs
y = scaler_y.fit_transform(df.iloc[:, 5:].values)  # Normalize outputs

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define Neural Network
class PerformanceNet(nn.Module):
    def __init__(self, input_size=5, hidden_size=10, output_size=8):
        super(PerformanceNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# Initialize model, loss, and optimizer
model = PerformanceNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 3000
for epoch in range(epochs):
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# Save model
torch.save(model.state_dict(), "performance_model.pth")
print("Model saved as performance_model.pth")

# Evaluate on test data
model.eval()
test_predictions = model(X_test)
test_loss = criterion(test_predictions, y_test)
print(f"Test Loss: {test_loss.item():.4f}")

# Example: Make a prediction on new data
X_sample = torch.tensor([[16.0, 7.0, 0.6, 6.0, 33.0]], dtype=torch.float32)
predictions = model(X_sample)
print("Predicted Performance Metrics:", predictions.detach().numpy())

