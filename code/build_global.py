import os
import pandas as pd

BASE = r"D:\Project_SOC_Kitsune\data"
OUT_DIR = r"D:\Project_SOC_Kitsune\global"
OUT_PARQUET = os.path.join(OUT_DIR, "kitsune_global.parquet")

PAIRS = [
    ("Active_Wiretap", "Active_Wiretap_dataset.csv", "Active_Wiretap_labels.csv"),
    ("Fuzzing", "Fuzzing_dataset.csv", "Fuzzing_labels.csv"),
    ("Mirai", "Mirai_dataset.csv", "mirai_labels.csv"),  # chez toi: labels en minuscule
    ("MitM", "MitM_dataset.csv", "MitM_labels.csv"),
    ("OS_Scan", "OS_scan_dataset.csv", "OS_Scan_labels.csv"),
    ("SSDP_Flood", "SSDP_Flood_dataset.csv", "SSDP_Flood_labels.csv"),
    ("SSL_Renegotiation", "SSL_Renegotiation_dataset.csv", "SSL_Renegotiation_labels.csv"),
    ("SYN_DoS", "SYN_DoS_dataset.csv", "SYN_DoS_labels.csv"),
    ("Video_Injection", "Video_injection_dataset.csv", "Video_Injection_labels.csv"),
]

N_FEATURES = 115

def load_features_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    if df.shape[1] == 116:
        # Mirai: 1ere colonne = index => à drop
        df = df.iloc[:, 1:]

    if df.shape[1] != N_FEATURES:
        raise ValueError(f"{os.path.basename(path)}: {df.shape[1]} colonnes (attendu {N_FEATURES})")

    df.columns = [f"f{i}" for i in range(N_FEATURES)]
    return df

def load_labels_csv(path: str) -> pd.Series:
    y = pd.read_csv(path, header=None).iloc[:, 0]
    y = y.astype(float).astype(int)  # 0/1
    return y

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    chunks = []
    for attack_name, ds_file, lb_file in PAIRS:
        ds_path = os.path.join(BASE, ds_file)
        lb_path = os.path.join(BASE, lb_file)

        if not os.path.exists(ds_path):
            raise FileNotFoundError(ds_path)
        if not os.path.exists(lb_path):
            raise FileNotFoundError(lb_path)

        X = load_features_csv(ds_path)
        y = load_labels_csv(lb_path)

        if len(X) != len(y):
            raise ValueError(f"{attack_name}: lignes mismatch X={len(X)} y={len(y)}")

        df = X.copy()
        df["label_binary"] = y.values
        df["attack_name"] = attack_name
        chunks.append(df)

        print(f"[OK] {attack_name}: X={X.shape}, y={y.shape}")

    global_df = pd.concat(chunks, ignore_index=True)
    global_df.to_parquet(OUT_PARQUET, index=False)

    print(f"\n✅ Saved: {OUT_PARQUET}")
    print("Columns:", len(global_df.columns))
    print(global_df.head(2))

if __name__ == "__main__":
    main()
