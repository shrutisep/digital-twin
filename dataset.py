import numpy as np
import pandas as pd

np.random.seed(42)  # For reproducibility

# Parameters
n_samples = 500
time = np.linspace(0, 1, n_samples)

# Helper to add waveform noise
def noisy_wave(amplitude, freq, base=0, std=0.05):
    return base + amplitude * np.sin(2*np.pi*freq*time) + np.random.normal(0, std, n_samples)

data_defs = {
    "Healthy": {
        "current": noisy_wave(1.0, 50, 1),
        "voltage": noisy_wave(220, 50),
        "temperature": 40 + np.random.normal(0, 1, n_samples)
    },
    "Rotor Fault": {
        "current": noisy_wave(1.2, 50, 1) + noisy_wave(0.2, 100),
        "voltage": noisy_wave(210, 50),
        "temperature": 45 + np.random.normal(0, 2, n_samples)
    },
    "Bearing Fault": {
        "current": noisy_wave(1.0, 50, 1) + noisy_wave(0.5, 200),
        "voltage": noisy_wave(220, 50),
        "temperature": 50 + np.random.normal(0, 2, n_samples)
    },
    "Overload": {
        "current": noisy_wave(1.6, 50, 2),
        "voltage": noisy_wave(205, 50),
        "temperature": 60 + np.random.normal(0, 3, n_samples)
    },
    "Stator Short": {
        "current": noisy_wave(1.8, 50, 2),
        "voltage": noisy_wave(180, 50),
        "temperature": 70 + np.random.normal(0, 4, n_samples)
    },
    "Voltage Sag": {
        "current": noisy_wave(1.0, 50, 1),
        "voltage": noisy_wave(120, 50),  # Simulate a sag in voltage
        "temperature": 40 + np.random.normal(0, 1, n_samples)
    }
}

# Generate and save datasets
for fault_type, signals in data_defs.items():
    df = pd.DataFrame({
        "time": time,
        "current": signals["current"],
        "voltage": signals["voltage"],
        "temperature": signals["temperature"]
    })
    df.to_csv(f"{fault_type.replace(' ', '_').lower()}.csv", index=False)
    print(f"{fault_type} dataset saved.")

