import torch
import pandas as pd
from datetime import timedelta
from torch import nn
import os
import torch
from torch.utils.tensorboard import SummaryWriter
import pickle

class power_Transformer(nn.Module):
    def __init__(self, input_dim, output_dim=48):
        super(power_Transformer, self).__init__()
        self.embedding = nn.Linear(feature_dim + 1, input_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, batch_first=True),
            num_layers=4
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x, mask=None):
        print(f"Input x shape: {x.shape}")
        print(f"Mask shape: {mask.shape}" if mask is not None else "Mask is None")
        batch_size, _, feature_dim = x.size()
        padding_token = torch.ones(batch_size, 1, feature_dim, device=x.device)
        x = torch.cat((padding_token, x), dim=1)
        print(f"x after adding padding_token shape: {x.shape}")
        x_embedding = self.embedding(x)
        print(f"x_embedding shape: {x_embedding.shape}")
        x = self.encoder(x_embedding, src_key_padding_mask=mask)
        print(f"Encoded x shape: {x.shape}")
        representation = x[:, 0, :]
        output = self.decoder(representation)
        print(f"Output shape: {output.shape}")
        return output

input_dim = 64
output_dim = 48
feature_dim = 12
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "/Users/chenruoting/Desktop/Course/DeepLearning/finalproject/aicup_model/power_transformer_model_epoch_1_step_2200.pth"
# 1200 1900 2200
model = power_Transformer(input_dim, output_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Model loaded from {model_path}")

    
def process_time_step(current_time, current_location):
    if isinstance(current_time, str):
        current_time = pd.to_datetime(current_time)
    past_data = data[(data['LocationCode'] == current_location) &
                     (data['DateTime'] < current_time)].tail(15)
    if len(past_data) < 15:
        padding = pd.DataFrame(0, index=range(15 - len(past_data)), columns=features + [target])
        past_data = pd.concat([padding, past_data])
    past_data = torch.tensor(past_data[features + [target]].values, dtype=torch.float32)
    print(f"Past data shape: {past_data.shape}")
    mask = [0]
    mask += [1] * (15 - len(past_data))
    mask += [0] * (len(past_data))
    other_stations = []
    for i in range(48):
        next_time_dic = station_data_cache.get(current_time + timedelta(minutes=10 * i), pd.DataFrame())
        if 'LocationCode' not in next_time_dic.columns:
            for _ in range(17):
                other_stations.append([0] * (feature_dim + 1))
                mask.append(1)
            continue
        filtered_data = next_time_dic[next_time_dic['LocationCode'] != current_location]
        x = 17 - len(filtered_data)
        station_data = filtered_data[features + [target]].values
        other_stations.extend(station_data)
        mask.extend([0] * len(station_data))
        other_stations.extend([[0] * (feature_dim + 1)] * x)
        mask.extend([1] * x)
    other_stations = torch.tensor(other_stations, dtype=torch.float32)
    print(f"Other stations shape: {other_stations.shape}")
    print(f"Mask shape: {len(mask)}")
    return past_data, other_stations, torch.tensor(mask, dtype=torch.float32)


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

    data = pd.read_csv('/Users/chenruoting/Desktop/Course/DeepLearning/finalproject/project/processed_data_10min_grouped.csv')
    data['DateTime'] = pd.to_datetime(data['DateTime'])

    data['day'] = data['DateTime'].dt.day
    data['month'] = data['DateTime'].dt.month
    data['hour'] = data['DateTime'].dt.hour

    features = ['LocationCode', 'WindSpeed(m/s)', 'Pressure(hpa)', 'Temperature(°C)', 'Humidity(%)', 'Sunlight(Lux)', 'Place', 'Height', 'Direction', 'hour', 'day', 'month']
    target = 'Power(mW)'

predictions = []
test_data = pd.read_csv('/Users/chenruoting/Desktop/Course/DeepLearning/finalproject/36_TestSet_SubmissionTemplate/filtered_test_data.csv')
for _, row in test_data.iterrows():
    print(row)
    current_time = row["DateTime"]
    current_location = row["LocationCode"]
    past_data, current_data, mask = process_time_step(current_time, current_location)
    past_data = past_data.unsqueeze(0).to(device)
    current_data = current_data.unsqueeze(0).to(device)
    mask = mask.unsqueeze(0).to(device)
    print(f"Sequence data shape before model: past_data {past_data.shape}, current_data {current_data.shape}, mask {mask.shape}")
    sequence_data = torch.cat((past_data, current_data), dim=1)
    print(f"Sequence data shape: {sequence_data.shape}")
    output = model(sequence_data, mask=mask)
    print(f"Model output shape: {output.shape}")
    predictions.append(output.detach().cpu().numpy())

test_data['Predictions'] = [list(pred) for pred in predictions]
output_file = "/Users/chenruoting/Desktop/Course/DeepLearning/finalproject/36_TestSet_SubmissionTemplate/predictions.csv"
test_data.to_csv(output_file, index=False)
print(f"Predictions saved to {output_file}")
