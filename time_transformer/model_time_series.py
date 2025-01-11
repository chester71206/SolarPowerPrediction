from datetime import timedelta
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
cache_file = './time_series_cache/station_data_cache.pkl'
os.makedirs(os.path.dirname(cache_file), exist_ok=True)

if os.path.exists(cache_file):
    with open(cache_file, 'rb') as f:
        station_data_cache = pickle.load(f)
    print("save cache!")
    model_dir = './time_series_model_log'
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='./logs')
    batch_size = 64
    sequence_length = 0  # sequence_length = n_past + (num_stations - 1) + 1
    feature_dim = 12  # 
    num_stations = 17  # 總共有 17 個發電站

    data = pd.read_csv('/mnt/disk2/kuan/time_transformer/processed_data_10min_grouped.csv')
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    data['day'] = data['DateTime'].dt.day
    data['month'] = data['DateTime'].dt.month
    data['hour'] = data['DateTime'].dt.hour

    features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Place', 'Height', 'Direction', 'hour', 'day', 'month']
    target = 'Power(mW)'
else:
    model_dir = './time_series_model_log'
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir='./logs')
    batch_size = 64
    sequence_length = 0  # sequence_length = n_past + (num_stations - 1) + 1
    feature_dim = 12  # 
    num_stations = 17  # 總共有 17 個發電站

    data = pd.read_csv('/mnt/disk2/kuan/time_transformer/processed_data_10min_grouped.csv')
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    data['day'] = data['DateTime'].dt.day
    data['month'] = data['DateTime'].dt.month
    data['hour'] = data['DateTime'].dt.hour

    features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Place', 'Height', 'Direction', 'hour', 'day', 'month']
    target = 'Power(mW)'

    # 用 DateTime 分组，以 LocationCode 為 index 建立 cache
    """
    {
        Timestamp('2024-11-08 12:00:00'): 
            WindSpeed(m/s)  Power(mW)
        LocationCode                      
        1                3.4        100
        2                2.5        150
        3                4.1        200,
        
        Timestamp('2024-11-08 12:01:00'): 
            WindSpeed(m/s)  Power(mW)
        LocationCode                      
        1                3.6        110
        2                2.3        140
        3                4.2        210
    }

    """
    data_grouped = data.groupby('DateTime')
    station_data_cache = {time: group.reset_index() for time, group in data_grouped}

    with open(cache_file, 'wb') as f:
        pickle.dump(station_data_cache, f)
        print("cache save!")

class PowerDataset(Dataset):
    def __init__(self, data, n_past=15):
        self.data = data
        self.n_past = n_past  # 前 n 個 time step
        self.unique_times = data['DateTime']
        self.feature_dim = len(features)
    
    def __len__(self):
        return len(self.unique_times)
    
    def __getitem__(self, idx):
        features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Place', 'Height', 'Direction', 'hour', 'day', 'month']
        target = 'Power(mW)'
        mask = [0]
        current_time = self.unique_times.iloc[idx]
        current_location = self.data['LocationCode'].iloc[idx]
        # print(current_time)
        # print(current_location)

        current_time_dic = station_data_cache.get(current_time, pd.DataFrame())
        # print(current_time)
        # print(current_time_dic.head())
        # print(current_time_dic.columns)
        # print(current_time_dic) # 跟 past data 無關
        # 選擇前 n_past 的目標發電站資料
        past_data = self.data[(self.data['LocationCode'] == current_location) & 
                            (self.data['DateTime'] < current_time)].tail(self.n_past)

        # print("past_data: ", past_data)
        # 如果不足 n_past，用 0 input
        if len(past_data) < self.n_past:
            padding = pd.DataFrame(0, index=range(self.n_past - len(past_data)), columns=features + [target])
            past_data = pd.concat([padding, past_data])
        
        mask += [1] * (self.n_past - len(past_data))
        mask += [0] * (len(past_data))

        # 提取其他發電站在當前時間的資料，排除目標發電站
        other_stations = []
        # location_codes = current_time_dic['LocationCode'].unique().tolist()
        # print("LocationCode list:", location_codes)
        # for station_id in range(1, 18): # 1 到 17
        # if current_time_dic['LocationCode'] == current_location:
        #     other_stations.append([0] * (self.feature_dim + 1))
        #     mask.append(1)
        for i in range(0,48):
            next_time_dic = station_data_cache.get(current_time + timedelta(minutes=10*i), pd.DataFrame())
            if 'LocationCode' not in next_time_dic.columns:
                # print(f"'LocationCode' not found in columns for time: {current_time + timedelta(minutes=10 * i)}")
                # print(f"Available columns: {next_time_dic.columns}")
                for _ in range(17):
                    other_stations.append([0] * (self.feature_dim + 1))
                    mask.append(1)
                continue
            filtered_data = next_time_dic[next_time_dic['LocationCode']!= current_location]
            # print("Filtered data columns:", filtered_data.columns)
            # print(filtered_data)
            x = 17 - filtered_data.shape[0]

            station_data = filtered_data[features + [target]].values
            for data in station_data:
                other_stations.append(data)
                mask.append(0)

            for _ in range(x):
                other_stations.append([0] * (self.feature_dim + 1))
                mask.append(1)
        
        future_data = self.data[(self.data['LocationCode'] == current_location) &
                            (self.data['DateTime'] >= current_time)].head(48)
        if len(future_data) < 48:
            padding = pd.DataFrame(0, index=range(48 - len(future_data)), columns=[target])
            future_data = pd.concat([future_data, padding])

        future_targets = future_data[target].values  # [48]
        # print(f"Past data shape before tensor conversion: {past_data.shape}")
        # print(f"Past data columns: {past_data.columns}")
        past_data = torch.tensor(past_data[features + [target]].values, dtype=torch.float32)  # [n_past, feature_dim + 1]
        # print(f"Past data Tensor shape: {past_data.shape}")
        current_data = torch.tensor(other_stations, dtype=torch.float32)  # [16, feature_dim + 1]
        mask = torch.tensor(mask, dtype=torch.float32)  # [16]
        future_targets = torch.tensor(future_targets, dtype=torch.float32)  # [48]

        return future_targets, past_data, current_data, mask
    
# Past data Tensor shape: torch.Size([15, 13])
# past_data shape: torch.Size([64, 15, 13])
# current_data shape: torch.Size([64, 816, 13]) 48*17
# mask shape: torch.Size([64, 832])
# target shape: torch.Size([64, 48])

        # other_stations_data = self.data[(self.data['DateTime'] == current_time) & (self.data['LocationCode'] != current_location)]
        # other_stations = []

        # for station_id in range(1, num_stations + 1):
        #     if station_id == past_data['LocationCode'].iloc[-1]:  # 跳過目標發電站
        #         continue
        #     station_data = other_stations_data[other_stations_data['LocationCode'] == station_id]
        #     if station_data.empty:
        #         other_stations.append([0] * (self.feature_dim + 1))  # 用 0 填充，包含 Power(mW)
        #         mask.append(0)  # 設置 mask 為 1
        #     else:
        #         # 加上 Power(mW) 的值
        #         # print(features + [target])
        #         other_stations.append(station_data[features + [target]].values[0])
        #         mask.append(1)  # 有數據，mask 設置為 0

train_dataset = PowerDataset(data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定義 Transformer 模型
class power_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim=48):
        super(power_Transformer, self).__init__()
        self.embedding = nn.Linear(feature_dim + 1, input_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True),
            num_layers=8
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, mask=None):
        # 加入一個全 1 的 token，在序列的開頭
        batch_size, _, feature_dim = x.size()
        padding_token = torch.ones(batch_size, 1, feature_dim, device=x.device)
        x = torch.cat((padding_token, x), dim=1)  # [batch_size, 20, feature_dim + 1]
        # print("add token: ", x)
        # print(x.size())
        x_embedding = self.embedding(x)
        x = self.encoder(x_embedding, src_key_padding_mask=mask)
        
        # 提取第一個 token 作為 representation
        representation = x[:, 0, :]  # [batch_size, feature_dim + 1]
        output = self.decoder(representation)
        # print(f"Decoder output shape: {output.shape}")
        return output

input_dim = 64
output_dim = 48
model = power_Transformer(input_dim, output_dim).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.5)

log_interval = 10
num_epochs = 2
step = 0
for epoch in range(num_epochs):
    print("train start")
    model.train()
    epoch_loss = 0
    
    for future_targets, past_data, current_data, mask in train_loader:
        optimizer.zero_grad()
        future_targets = future_targets.to(device)
        past_data = past_data.to(device)
        current_data = current_data.to(device)
        mask = mask.to(device)
        # print("past_data shape:", past_data.shape)
        # print("current_data shape:", current_data.shape)
        # print("mask shape:", mask.shape)
        # print("target shape:", future_targets.shape)

        # 將 past_data 和 current_data 變成 sequence_data
        sequence_data = torch.cat((past_data, current_data), dim=1)# [batch_size, 19 -> 20, feature_dim + 1]
        # print("sequence shape:", sequence_data.shape)
        outputs = model(sequence_data, mask=mask)
        # print(f"outputs shape: {outputs.shape}")
        # print(f"future_targets shape: {future_targets.shape}")

        loss = torch.mean(torch.abs(outputs.squeeze() - future_targets))

        loss.backward()
        optimizer.step()
        if step % 8000 == 0 and step > 0:
            scheduler.step()
        
        if step % log_interval == 0:
            writer.add_scalar('Loss/step', loss.item(), step)
            print(f"Step {step}: Loss {loss.item()}")
        
        step += 1
        if step >= 2000 and step % 1000 == 0:
            model_path = os.path.join(model_dir, f'power_transformer_model_step_{step}.pth')
            torch.save(model.state_dict(), f'power_transformer_model_step_{step}.pth')
            print(f"model is saved to power_transformer_model_step_{step}.pth")

        epoch_loss += loss.item()
        print(f'Step: {step}')
    # 16500 step = 1 epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_epoch_loss, epoch)
    print(f'Epoch {epoch + 1}, Loss: {avg_epoch_loss}')
    

model_path = os.path.join(model_dir, 'power_transformer_model_final.pth')
torch.save(model.state_dict(), model_path)
print(f"Final model is saved to {model_path}")

writer.close()