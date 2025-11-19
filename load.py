import pandas as pd
import matplotlib.pyplot as plt

# List your CSV files
files = ["healthy.csv", "rotor_fault.csv", "bearing_fault.csv", "overload.csv", "stator_short.csv", "voltage_sag.csv"]

# Read, label, and plot only first 5 rows
for fname in files:
    df = pd.read_csv(fname)
    label = fname.split(".")[0].replace("_", " ").title()
    df["label"] = label

    # Select the first 5 samples
    subset = df.iloc[:5]

    plt.figure(figsize=(12,4))
    
    # Plot Current
    plt.subplot(1, 3, 1)
    plt.plot(subset['time'], subset['current'], marker='o', color='b')
    plt.title(f"{label} - Current (first 5 points)")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")

    # Plot Voltage
    plt.subplot(1, 3, 2)
    plt.plot(subset['time'], subset['voltage'], marker='o', color='g')
    plt.title(f"{label} - Voltage (first 5 points)")
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")

    # Plot Temperature
    plt.subplot(1, 3, 3)
    plt.plot(subset['time'], subset['temperature'], marker='o', color='r')
    plt.title(f"{label} - Temperature (first 5 points)")
    plt.xlabel("Time [s]")
    plt.ylabel("Temperature [°C]")

    plt.tight_layout()
    plt.savefig(f"{label.replace(' ','_').lower()}_first5_waveforms.png")
    plt.show()
