import pandas as pd
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error
# Step 1: 讀取所有檔案並將它們載入至字典中，每個地點一個DataFrame
file_paths = glob.glob("/mnt/disk2/kuan/new_L*_Train.csv")  # 請替換為實際路徑
location_data = {f"L{idx + 1}": pd.read_csv(file) for idx, file in enumerate(file_paths)}

# 合併所有地點的數據
all_data = pd.concat(location_data.values(), ignore_index=True)

# 數據預處理
all_data['DateTime'] = pd.to_datetime(all_data['DateTime'], format='mixed')

for col in ['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']:
    all_data[col] = pd.to_numeric(all_data[col], errors='coerce')
all_data.ffill(inplace=True)
all_data.bfill(inplace=True)

# MinMaxScaler()數據
scaler = MinMaxScaler()
all_data[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']] = scaler.fit_transform(
    all_data[['WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)']]
)

# One-Hot 編碼 LocationCode
encoder = OneHotEncoder(sparse_output=False)
location_codes_encoded = encoder.fit_transform(all_data[['LocationCode']])
location_code_columns = encoder.get_feature_names_out(['LocationCode'])
location_codes_df = pd.DataFrame(location_codes_encoded, columns=location_code_columns, index=all_data.index)
all_data = pd.concat([all_data, location_codes_df], axis=1)

# 初始化 X_train 和 y_train
X_train = []
y_train = []

# 對每個 LocationCode 的數據進行處理
for location, location_data in all_data.groupby('LocationCode'):
    location_data = location_data.sort_values(by='DateTime')

    # 每 10 個時間步做一次平均
    averaged_data = []
    for i in range(0, len(location_data), 10):
        subset = location_data.iloc[i:i+10]  # 每 10 步分成一組
        averaged_data.append(subset.mean(numeric_only=True))  # 計算每組的平均
    averaged_data = pd.DataFrame(averaged_data)

    # 檢查是否有足夠的步長來生成訓練樣本
    if len(averaged_data) > 30 + 48:  # 確保有足夠的數據進行 12 步輸入和 48 步輸出
        for i in range(len(averaged_data) - 30 - 48):
            # 提取前 12 步的特徵作為輸入
            features = averaged_data.iloc[i:i+18][[
                'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)',
                'Humidity(%)', 'Sunlight(Lux)', 'Power(mW)',
                'LocationCode_1.0', 'LocationCode_2.0', 'LocationCode_3.0', 'LocationCode_4.0',
                'LocationCode_5.0', 'LocationCode_6.0', 'LocationCode_7.0', 'LocationCode_8.0',
                'LocationCode_9.0', 'LocationCode_10.0', 'LocationCode_11.0', 'LocationCode_12.0',
                'LocationCode_13.0', 'LocationCode_14.0', 'LocationCode_15.0', 'LocationCode_16.0',
                'LocationCode_17.0']].values
            X_train.append(features)

            # 提取未來 48 步的 Power(mW) 作為目標
            target = averaged_data.iloc[i+30:i+30+48]['Power(mW)'].values
            y_train.append(target)

# 轉換成 NumPy 陣列
X_train = np.array(X_train)
y_train = np.array(y_train)

# 分割為訓練集、驗證集和測試集
X_train_val, X_test, y_train_val, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_train_final, X_val, y_train_final, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = TimeSeriesDataset(X_train_final, y_train_final)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# ===========================
# Step 3: Define the Model
# ===========================
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take the output of the last LSTM step
        return self.fc(out)


input_size = X_train.shape[2]
hidden_size = 256
output_size = 48
model = LSTMModel(input_size, hidden_size, output_size)

# ===========================
# Step 4: Training the Model
# ===========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
training_loss = []
validation_loss = []
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    training_loss.append(epoch_train_loss)


    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    epoch_val_loss = val_loss / len(val_loader)
    validation_loss.append(epoch_val_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

# ===========================
# Step 5: Evaluating the Model
# ===========================
model.eval()
test_loss = 0.0
predictions = []
true_values = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        predictions.append(outputs.cpu().numpy())
        true_values.append(targets.cpu().numpy())

print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# ===========================
# Step 6: Plot Results
# ===========================
predictions = np.concatenate(predictions, axis=0)
true_values = np.concatenate(true_values, axis=0)

# 反標準化
true_values_expanded = np.zeros((true_values.shape[0] * 48, 6))
true_values_expanded[:, -1] = true_values.flatten()

predictions_expanded = np.zeros((predictions.shape[0] * 48, 6))
predictions_expanded[:, -1] = predictions.flatten()

true_values_original = scaler.inverse_transform(true_values_expanded)[:, -1].reshape(true_values.shape)
predictions_original = scaler.inverse_transform(predictions_expanded)[:, -1].reshape(predictions.shape)

# Rescale Data
true_values_expanded = np.zeros((true_values.shape[0] * 48, 6))
true_values_expanded[:, -1] = true_values.flatten()

predictions_expanded = np.zeros((predictions.shape[0] * 48, 6))
predictions_expanded[:, -1] = predictions.flatten()


true_values_original = scaler.inverse_transform(true_values_expanded)[:, -1].reshape(true_values.shape)
predictions_original = scaler.inverse_transform(predictions_expanded)[:, -1].reshape(predictions.shape)
predictions_original = np.clip(predictions_original, a_min=0, a_max=None)

# Calculate MSE and MAE
mse = mean_squared_error(true_values_original.flatten(), predictions_original.flatten())
mae = mean_absolute_error(true_values_original.flatten(), predictions_original.flatten())
total_error = np.sum(np.abs(true_values_original.flatten() - predictions_original.flatten()))

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Total Error: {total_error:.4f}")

# Compare 50 actual and predicted values
comparison_data = pd.DataFrame({
    "Actual": true_values_original.flatten()[:1000],
    "Predicted": predictions_original.flatten()[:1000]
})
print("First 1000 Actual vs. Predicted values:")
print(comparison_data)

# Save the comparison to a CSV file for review
comparison_data.to_csv("comparison_data.csv", index=False)

# Plotting the results for the first 50 points
plt.figure(figsize=(12, 6))
plt.plot(true_values_original.flatten()[:1000], label="Actual", marker='o', linestyle="-", color="blue")
plt.plot(predictions_original.flatten()[:1000], label="Predicted", marker='x', linestyle="--", color="orange")
plt.legend()
plt.xlabel("Sample Index")
plt.ylabel("Power (mW)")
plt.title("Comparison of Actual and Predicted Values (First 1000 Samples)")
plt.grid(True)
plt.savefig("output_plot_1000_samples.png")
plt.show()




plt.figure(figsize=(10, 6))
plt.plot(training_loss, label="Training Loss", marker='o')
plt.plot(validation_loss, label="Validation Loss", marker='x')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Epochs")
plt.legend()
plt.grid(True)

# 儲存損失圖
plt.savefig("loss.png")
plt.show()
