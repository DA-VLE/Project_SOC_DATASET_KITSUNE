import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

# (optionnel) XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

DATA = r"D:\Project_SOC_Kitsune\data\global_dataset.csv"
OUTDIR = r"D:\Project_SOC_Kitsune\artifacts\v1"
os.makedirs(OUTDIR, exist_ok=True)

CHUNK = 200_000

# --- détecter colonnes
cols = pd.read_csv(DATA, nrows=0).columns.tolist()
cols = [c.strip() for c in cols]

FEATURES = [c for c in cols if c.startswith("f")]
if len(FEATURES) == 0:
    raise ValueError("Aucune colonne f0..f114 trouvée dans global_dataset.csv")

LABEL_COL = "label" if "label" in cols else None
ATTACK_COL = "attack_name" if "attack_name" in cols else None
if LABEL_COL is None:
    raise ValueError("Colonne 'label' introuvable dans global_dataset.csv")
if ATTACK_COL is None:
    raise ValueError("Colonne 'attack_name' introuvable dans global_dataset.csv")

print("features =", len(FEATURES), "label_col =", LABEL_COL, "attack_col =", ATTACK_COL)

# =========================================================
# 1) FIT SCALER sur tout le dataset (recommandé)
# =========================================================
scaler = StandardScaler()

for chunk in pd.read_csv(DATA, usecols=FEATURES + [LABEL_COL], chunksize=CHUNK):
    # features -> numeric
    X = chunk[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    if X.shape[0] == 0:
        continue
    scaler.partial_fit(X)

joblib.dump(scaler, os.path.join(OUTDIR, "scaler.joblib"))
print("[OK] scaler saved")

# =========================================================
# 2) IsolationForest sur BENIGN uniquement (sample)
# =========================================================
target_benign = 300_000
benign_sample = []
seen = 0

for chunk in pd.read_csv(DATA, usecols=FEATURES + [LABEL_COL], chunksize=CHUNK):
    y = pd.to_numeric(chunk[LABEL_COL], errors="coerce").fillna(1)  # NaN -> attaque
    benign = chunk[y == 0]
    if len(benign) == 0:
        continue

    Xb = benign[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(Xb).all(axis=1)
    Xb = Xb[mask]
    if Xb.shape[0] == 0:
        continue

    Xb = scaler.transform(Xb)

    take = min(Xb.shape[0], target_benign - seen)
    benign_sample.append(Xb[:take])
    seen += take
    if seen >= target_benign:
        break

if seen < 10_000:
    raise ValueError(f"Pas assez de benign pour IsolationForest (seen={seen}).")

X_benign = np.vstack(benign_sample)
print("benign sample:", X_benign.shape)

iso = IsolationForest(
    n_estimators=200,
    contamination=0.01,
    random_state=42,
    n_jobs=-1,
)
iso.fit(X_benign)
joblib.dump(iso, os.path.join(OUTDIR, "iso_forest.joblib"))
print("[OK] iso_forest saved")

# =========================================================
# 3) Dataset de classification (benign + attack type) via sampling
#    -> puis entraînement de 2 modèles: LR + (XGB ou HGB fallback)
# =========================================================
per_class = 80_000
counts = {}
X_list, y_list = [], []

rng = np.random.default_rng(42)

for chunk in pd.read_csv(DATA, usecols=FEATURES + [LABEL_COL, ATTACK_COL], chunksize=CHUNK):
    ybin = pd.to_numeric(chunk[LABEL_COL], errors="coerce").fillna(1).to_numpy()
    attack = chunk[ATTACK_COL].fillna("unknown").astype(str).to_numpy()

    # y en "benign" ou nom d'attaque
    y = np.where(ybin == 0, "benign", attack)

    # features clean
    X = chunk[FEATURES].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    mask = np.isfinite(X).all(axis=1)
    X = X[mask]
    y = y[mask]

    if X.shape[0] == 0:
        continue

    X = scaler.transform(X)

    for cls in np.unique(y):
        if counts.get(cls, 0) >= per_class:
            continue

        idx = np.where(y == cls)[0]
        if idx.size == 0:
            continue

        rng.shuffle(idx)  # évite de prendre toujours les premières lignes
        take = min(idx.size, per_class - counts.get(cls, 0))
        sel = idx[:take]

        X_list.append(X[sel])
        y_list.append(np.array([cls] * take, dtype=object))
        counts[cls] = counts.get(cls, 0) + take

if len(X_list) == 0:
    raise ValueError("Aucun échantillon collecté pour la classification.")

X_train = np.vstack(X_list)
y_train = np.concatenate(y_list)
classes = sorted(set(y_train.tolist()))
print("clf train:", X_train.shape, "classes:", len(classes))
print("classes:", classes)

# --- encoder labels (utile pour XGBoost + streaming cohérent)
le = LabelEncoder()
y_enc = le.fit_transform(y_train)

joblib.dump(le, os.path.join(OUTDIR, "label_encoder.joblib"))
print("[OK] label_encoder saved")

# ---------- (A) Logistic Regression ----------
lr = LogisticRegression(
    max_iter=2000,     # évite ConvergenceWarning
    solver="saga",     # ok pour multi-class + grand dataset
    n_jobs=-1
)
lr.fit(X_train, y_enc)
joblib.dump(lr, os.path.join(OUTDIR, "attack_classifier_lr.joblib"))
print("[OK] attack_classifier_lr saved")

# ---------- (B) XGBoost si dispo, sinon fallback HGB ----------
if HAS_XGB:
    n_classes = len(le.classes_)
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=n_classes,
        n_estimators=300,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42
    )
    xgb.fit(X_train, y_enc)
    joblib.dump(xgb, os.path.join(OUTDIR, "attack_classifier_xgb.joblib"))
    print("[OK] attack_classifier_xgb saved")
else:
    hgb = HistGradientBoostingClassifier(
        max_depth=10,
        learning_rate=0.1,
        max_iter=300,
        random_state=42
    )
    hgb.fit(X_train, y_enc)
    joblib.dump(hgb, os.path.join(OUTDIR, "attack_classifier_hgb.joblib"))
    print("[OK] attack_classifier_hgb saved (fallback, xgboost not installed)")

print("[DONE] artifacts in", OUTDIR)
