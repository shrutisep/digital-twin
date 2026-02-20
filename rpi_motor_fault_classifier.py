#!/usr/bin/env python3
"""
Raspberry Pi - receives CSV lines from STM32 via UART
runs lightweight ML model → classifies fault
"""

import serial
import csv
import io
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────
SERIAL_PORT     = "/dev/ttyUSB0"          # change to your port
BAUDRATE        = 921600
MODEL_PATH      = "rf_motor_fault_model.pkl"   # your trained model
FEATURE_COLS    = ["slip", "speed_rpm", "Te_Nm", "i_a", "i_b", "i_c", "i_ds", "i_qs"]
CLASS_NAMES     = ["healthy", "broken_bar", "stator_short", "eccentricity"]

# ───────────────────────────────────────────────
# Load pre-trained model (you must train & save it first)
# ───────────────────────────────────────────────
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# ───────────────────────────────────────────────
# Open serial port
# ───────────────────────────────────────────────
try:
    ser = serial.Serial(
        port=SERIAL_PORT,
        baudrate=BAUDRATE,
        timeout=1
    )
    print(f"Connected to {SERIAL_PORT} @ {BAUDRATE} baud")
except Exception as e:
    print(f"Cannot open serial port: {e}")
    exit(1)

# Skip header line if present
ser.readline()

# Buffer for sliding window (if your model needs several samples)
window_size = 50          # e.g. 50 samples = 250 ms @ 200 Hz
window = []

print("Listening for motor data... (Ctrl+C to exit)\n")

try:
    while True:
        line = ser.readline().decode('utf-8', errors='ignore').strip()

        if not line:
            continue

        # Parse CSV line
        try:
            reader = csv.reader(io.StringIO(line))
            row = next(reader)
            if len(row) < len(FEATURE_COLS) + 1:  # +1 for time
                continue

            # Extract features (skip time column)
            features = [float(x) for x in row[1:1+len(FEATURE_COLS)]]
            window.append(features)

            if len(window) >= window_size:
                # Use last window_size samples or average / flatten
                X = np.array(window[-window_size:]).flatten()   # or .mean(axis=0)

                # Predict
                pred = model.predict([X])[0]
                prob = model.predict_proba([X])[0]

                fault = CLASS_NAMES[pred]
                confidence = prob[pred] * 100

                print(f"[{time.strftime('%H:%M:%S')}] Fault: {fault:12}  "
                      f"conf: {confidence:5.1f}%   slip:{float(row[1]):6.4f}")

                # Optional: act on result
                if fault != "healthy" and confidence > 85:
                    print("  ⚠️  FAULT DETECTED — take action!")

                # Keep window rolling (overlap)
                window = window[-window_size//2:]

        except (ValueError, IndexError, csv.Error):
            # bad line — skip
            pass

        time.sleep(0.001)  # small sleep to not overload CPU

except KeyboardInterrupt:
    print("\nStopped by user")

finally:
    ser.close()
    print("Serial port closed")
