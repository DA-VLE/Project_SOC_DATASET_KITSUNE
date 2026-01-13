import os
import numpy as np
import pandas as pd

DATA_DIR = r"D:\Project_SOC_Kitsune\data"
OUT_CSV  = os.path.join(DATA_DIR, "global_dataset.csv")

# (dataset_file, labels_file, attack_name)
SOURCES = [
    ("Mirai_dataset.csv",               "Mirai_labels.csv",               "Mirai"),
    ("Fuzzing_dataset.csv",             "Fuzzing_labels.csv",             "Fuzzing"),
    ("OS_Scan_dataset.csv",             "OS_Scan_labels.csv",             "OS_Scan"),
    ("SYN_DoS_dataset.csv",             "SYN_DoS_labels.csv",             "SYN_DoS"),
    ("MitM_dataset.csv",                "MitM_labels.csv",                "MitM"),
    ("Video_Injection_dataset.csv",     "Video_Injection_labels.csv",     "Video_Injection"),
    ("SSL_Renegotiation_dataset.csv",   "SSL_Renegotiation_labels.csv",   "SSL_Renegotiation"),
    ("SSDP_Flood_dataset.csv",          "SSDP_Flood_labels.csv",          "SSDP_Flood"),
    ("Active_Wiretap_dataset.csv",      "Active_Wiretap_labels.csv",      "Active_Wiretap"),
]

N_FEATURES_TARGET = 115  # f0..f114

import pandas as pd
import numpy as np

def load_labels_csv(path: str) -> pd.Series:
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        first = f.readline().strip()

    # --- Cas A: Mirai (1 colonne, pas de virgule, pas de header)
    if "," not in first:
        df = pd.read_csv(path, header=None, encoding="utf-8-sig", dtype=str)
        s = df.iloc[:, 0].astype(str).str.strip()
        s = s[s != ""]                 # enlève lignes vides
        y = pd.to_numeric(s, errors="raise").astype(np.int8)

        if not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"[LABELS] valeurs !=0/1 dans {path}: {sorted(y.unique())}")

        return y.reset_index(drop=True)

    # --- Cas B: labels Kitsune avec header "","x" (2 colonnes)
    df = pd.read_csv(path, encoding="utf-8-sig")

    if "x" in df.columns:
        s = df["x"]
    else:
        s = df.iloc[:, -1]

    s = s.astype(str).str.strip()
    s = s[s != ""]
    y = pd.to_numeric(s, errors="raise").astype(np.int8)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"[LABELS] valeurs !=0/1 dans {path}: {sorted(y.unique())}")

    return y.reset_index(drop=True)


def normalize_features_shape(df: pd.DataFrame, src_name: str) -> pd.DataFrame:
    """
    Assure qu'on obtient exactement 115 features.
    - Si 116 colonnes: supprime la 1ère (index Mirai)
    - Si 115: ok
    - Sinon: erreur
    """
    n = df.shape[1]
    if n == N_FEATURES_TARGET:
        return df
    if n == N_FEATURES_TARGET + 1:
        # drop first column (index)
        return df.iloc[:, 1:]
    raise ValueError(f"[{src_name}] Nombre de colonnes inattendu: {n} (attendu 115 ou 116)")

def main():
    # Recrée le fichier global
    if os.path.exists(OUT_CSV):
        os.remove(OUT_CSV)
    first_write = True
    chunksize = 50_000  # ajuste si besoin (20k si PC faible)

    for ds_file, lb_file, attack_name in SOURCES:
        ds_path = os.path.join(DATA_DIR, ds_file)
        lb_path = os.path.join(DATA_DIR, lb_file)

        if not os.path.exists(ds_path):
            raise FileNotFoundError(ds_path)
        if not os.path.exists(lb_path):
            raise FileNotFoundError(lb_path)
        
        print(f"\n=== {attack_name} ===")
        print(f"dataset: {ds_path}")
        print(f"labels : {lb_path}")
        y = load_labels_csv(lb_path)
        y_len = len(y)
        print(f"labels rows = {y_len}")
        start = 0
        total_rows = 0
        # Lecture en chunks
        for chunk in pd.read_csv(
            ds_path,
            header=None,
            engine="c",
            dtype=np.float32,        # + rapide, + léger
            chunksize=chunksize
        ):
            chunk = normalize_features_shape(chunk, attack_name)

            # renomme f0..f114
            chunk.columns = [f"f{i}" for i in range(N_FEATURES_TARGET)]

            end = start + len(chunk)
            if end > y_len:
                raise ValueError(f"[{attack_name}] Dataset a plus de lignes que labels (end={end} > y={y_len})")
            chunk["label"] = y[start:end]
            chunk["attack_name"] = attack_name

            # écriture append
            chunk.to_csv(OUT_CSV, mode="a", index=False, header=first_write)
            first_write = False
            start = end
            total_rows += len(chunk)

            if total_rows % (chunksize * 5) == 0:
                print(f"  wrote {total_rows} rows...")

        if start != y_len:
            raise ValueError(f"[{attack_name}] Labels a plus de lignes que dataset (dataset={start}, labels={y_len})")
        print(f"OK: {attack_name} -> {total_rows} rows written")
    print(f"\n✅ Global dataset écrit: {OUT_CSV}")

if __name__ == "__main__":
    main()
