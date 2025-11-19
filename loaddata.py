import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load your STM32 data
df_live = pd.read_csv("stm32values.log", header=None, names=["current", "voltage", "temperature"])
print("STM32 data preview:")
print(df_live.head())

# 2. Load your labeled simulation datasets for training the classifier
simulation_files = ["healthy.csv", "rotor_fault.csv", "bearing_fault.csv", "overload.csv", "stator_short.csv", "voltage_sag.csv"]

dfs = []
for fname in simulation_files:
    sim_df = pd.read_csv(fname)
    label = fname.split(".")[0].replace("_", " ").title()  # E.g.: 'Healthy', 'Rotor Fault'
    sim_df["label"] = label
    dfs.append(sim_df)

# Concatenate all simulation data
data = pd.concat(dfs, ignore_index=True)
X = data[["current", "voltage", "temperature"]]
y = data["label"]

# 3. Train/test split and fit classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Predict fault type for each STM32 sample
df_live['predicted_fault'] = model.predict(df_live[["current", "voltage", "temperature"]])
print("Live STM32 predictions preview:")
print(df_live.head())

# 5. Plot the STM32 waveforms with detected fault types
plt.figure(figsize=(12,6))
plt.plot(df_live['current'], label='Current [A]')
plt.plot(df_live['voltage'], label='Voltage [V]')
plt.plot(df_live['temperature'], label='Temperature [°C]')
plt.title("STM32 Motor Signal Waveforms")
plt.xlabel("Sample (time index)")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

# (Optional) Visualize predicted fault as color blocks
plt.figure(figsize=(12,2))
plt.scatter(range(len(df_live)), df_live['predicted_fault'].astype('category').cat.codes, 
            c=df_live['predicted_fault'].astype('category').cat.codes, cmap='tab10', s=10)
plt.yticks(ticks=range(len(df_live['predicted_fault'].unique())), labels=df_live['predicted_fault'].unique())
plt.xlabel("Sample (time index)")
plt.title("Predicted Fault Type (per STM32 sample)")
plt.tight_layout()
plt.show()
