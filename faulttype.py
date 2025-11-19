import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load and label as before
files = ["healthy.csv", "rotor_fault.csv", "bearing_fault.csv", "overload.csv", "stator_short.csv", "voltage_sag.csv"]

dfs = []
for fname in files:
    df = pd.read_csv(fname)
    label = fname.split(".")[0].replace("_", " ").title()
    df["label"] = label
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)
X = data.drop(['label', 'time'], axis=1)
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User input
print("\nEnter the values for your sample:")
time_in = float(input("Time [in seconds]: "))
current_in = float(input("Current [A]: "))
voltage_in = float(input("Voltage [V]: "))
temperature_in = float(input("Temperature [°C]: "))
user_sample = pd.DataFrame([{
    "current": current_in,
    "voltage": voltage_in,
    "temperature": temperature_in
}])
predicted_fault = model.predict(user_sample)[0]
print(f"\nPredicted Fault Type: {predicted_fault}")

ref_df = data[data['label'] == predicted_fault]

means = ref_df[['current', 'voltage', 'temperature']].mean()
your_values = [current_in, voltage_in, temperature_in]
plt.figure(figsize=(7,5))
bar_labels = ['Current', 'Voltage', 'Temperature']
plt.bar(bar_labels, means, color=['b','g','r'], alpha=0.6, label='Dataset Mean')
plt.bar(bar_labels, your_values, color=['b','g','r'], alpha=1, label='Your Sample', width=0.3)
plt.title(f"{predicted_fault} - Your Sample vs Typical Mean")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.show()

