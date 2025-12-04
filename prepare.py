import pandas as pd
import numpy as np
import os

DATA_DIR = "data"
OUT_DIR = "prepared"
os.makedirs(OUT_DIR, exist_ok=True)

def prepare_dataset(csv_name):
    path = os.path.join(DATA_DIR, csv_name)
    print(f"\nLoading {path}")

    df = pd.read_csv(path)
    print("Original shape:", df.shape)

    # ---- Detect timestamp column ----
    if "Unnamed: 0" in df.columns:
        time_col = "Unnamed: 0"
        print("Detected timestamp column: Unnamed: 0")
    else:
        # fallback
        time_col = df.columns[0]
        print(f"Using first column as timestamp: {time_col}")

    # ---- Rename ----
    df = df.rename(columns={time_col: "timestamp"})

    # ---- Convert timestamp ----
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ---- Sort ----
    df = df.sort_values("timestamp").reset_index(drop=True)

    # ---- Sensor columns ----
    sensor_cols = [c for c in df.columns if c != "timestamp"]
    print(f"Detected {len(sensor_cols)} sensors")

    # ---- Fill missing values ----
    df[sensor_cols] = df[sensor_cols].interpolate().bfill().ffill()

    # ---- Safety check ----
    assert df.isna().sum().sum() == 0, "Error: NaNs still present!"

    # ---- Save cleaned CSV ----
    out_csv = os.path.join(OUT_DIR, csv_name.replace(".csv", "_clean.csv"))
    df.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv} | shape = {df.shape}")

    # ---- Save values for ML models ----
    np.save(os.path.join(OUT_DIR, csv_name.replace(".csv", "_values.npy")),
            df[sensor_cols].values)

    np.save(os.path.join(OUT_DIR, csv_name.replace(".csv", "_timestamps.npy")),
            df["timestamp"].values)

    print("Exported NumPy arrays ✔")

prepare_dataset("METR-LA.csv")
prepare_dataset("PEMS-BAY.csv")

print("\n🎉 Done! Cleaned datasets are ready.")
