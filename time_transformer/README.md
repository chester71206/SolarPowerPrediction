# **AI CUP 2024 Power Generation Prediction**

This project is developed for **AI CUP 2024 Competition** to predict power generation based on **regional microclimate data**. Below is a detailed explanation of how the model is designed, the preprocessing steps, and how predictions are generated.

---

## **Overview**
- **Objective**: Predict the power generation for a specific location over 48 timesteps (10-minute intervals) using historical data and regional microclimate features.
- **Model Architecture**: A custom **Transformer-based model**.
- **Input Structure**:
  - **Past Data**: Historical data for the target location.
  - **Current Data**: Microclimate data from other locations.
  - **Mask**: Indicates missing or invalid data.
- **Output**: Predicted power generation for 48 timesteps (10 minutes each, total 8 hours).

---

## **Data Preprocessing**
### **Original Data Format**
The original dataset contains **minute-level data**, which was rearranged into **10-minute units** for prediction, aligning with the competition requirements.

### **Preprocessed Features**
The input features for each timestep include:
1. `LocationCode`: Unique identifier for each location.
2. `WindSpeed(m/s)`
3. `Pressure(hpa)`
4. `Temperature(°C)`
5. `Humidity(%)`
6. `Sunlight(Lux)`
7. `Place`
8. `Height`
9. `Direction`
10. `hour`
11. `day`
12. `month`
13. `Power(mW)` (target variable).

### **Target Data**
- Prediction is performed for a **specific location** and **time range**:
  - Target: `2024-01-17 09:00~17:00`, LocationCode: `01`.
  - Output: Power generation over 48 timesteps.

---

## **Model Design**
### **Transformer-Based Model**
The model utilizes a Transformer encoder-decoder structure to capture dependencies in the data:
- **Input Components**:
  - **Past Data**: The past 15 timesteps for the target location.
    - Shape: `torch.Size([1, 15, 13])`
  - **Current Data**: Microclimate data from all locations at the current and future timesteps.
    - Shape: `torch.Size([1, 816, 13])`
    - `816` = `48` timesteps × `17` locations.
  - **Mask**: Used to indicate missing or invalid data.
    - Shape: `torch.Size([1, 831])`.

### **Handling Missing Data**
- If data is missing for a specific location or timestep, the model fills the values with `0` and sets the corresponding **mask** value to `1`. This ensures the model does not use these invalid inputs in the attention mechanism.

### **Padding**
- A special **padding token** is added to the input sequence, increasing the shape to:
  - Sequence: `torch.Size([1, 832, 13])`.
  - Mask: `torch.Size([1, 832])`.

---

## **Example Input and Output**

### **Input**
1. **Past Data** (15 timesteps, 13 features for the target location):
   ```plaintext
   LocationCode, DateTime, WindSpeed, Pressure, Temperature, Humidity, Sunlight, Place, Height, Direction, hour, day, month, Power
   1, 2024-01-17 06:20:00, ..., ..., ..., ..., ..., ..., ..., ..., ..., ..., ...
   1, 2024-01-17 06:30:00, ..., ..., ..., ..., ..., ..., ..., ..., ..., ..., ...
   ...
2. **Current Data**:
Includes 48 timesteps across 17 locations.
Missing data is filled with 0.
3. **Mask**:
Indicates valid (0) or invalid (1) entries.

### **Output**
Predicted power values for the target location over 48 timesteps:
Time: 2024-01-17 09:00:00 - 2024-01-17 17:00:00
Predicted Power (mW): [123.45, 456.78, ..., 789.01]

---
## Model Workflow

### Embedding:
Input dimensions: [batch_size, sequence_length, feature_dim].
Embedding dimensions: [batch_size, sequence_length, input_dim].
### Transformer Encoder:
Uses multi-head attention to capture dependencies across locations and timesteps.
### Output Representation:
Extracts the first token's embedding to represent the output sequence:
representation = x[:, 0, :].
### Decoder:
Decodes the representation into the predicted power values.