"""Étape 4 : entraîner le détecteur d'anomalies.

Entrées : data/features_train.csv, data/features_test.csv
Sorties : model/autoencoder.keras, model/threshold.json, model/erreurs_test.png
"""
import json
import os
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                     # pas de fenêtre : on enregistre l'image
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras

RACINE = Path(__file__).resolve().parent
DATA, MODEL = RACINE / "data", RACINE / "model"

np.random.seed(42)
tf.random.set_seed(42)
MODEL.mkdir(exist_ok=True)

# ---------------------------------------------------------------- données
train = pd.read_csv(DATA / "features_train.csv")
test = pd.read_csv(DATA / "features_test.csv")
NOMS = [c for c in train.columns if c != "label"]

X_train_all = train[NOMS].to_numpy(dtype="float32")   # uniquement du normal
X_test = test[NOMS].to_numpy(dtype="float32")
y_test = test["label"].to_numpy()

# 80 % début pour apprendre, 20 % fin pour choisir le seuil (validation)
coupure = int(0.8 * len(X_train_all))
X_fit, X_val = X_train_all[:coupure], X_train_all[coupure:]


def choisir_seuil(scores_val, pourcentile=99):
    """Seuil = valeur dépassée par seulement 1 % des fenêtres normales."""
    return float(np.percentile(scores_val, pourcentile))


def evaluer(nom, scores_test, seuil):
    y_pred = (scores_test > seuil).astype(int)
    p = precision_score(y_test, y_pred, zero_division=0)
    r = recall_score(y_test, y_pred, zero_division=0)
    f = f1_score(y_test, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    print(f"\n=== {nom} ===")
    print(f"Précision : {p:.2f} | Rappel : {r:.2f} | F1 : {f:.2f}")
    print(f"Vrais normaux : {tn} | Fausses alertes : {fp} | "
          f"Anomalies ratées : {fn} | Anomalies détectées : {tp}")


# ------------------------------------------- 1) base de comparaison simple
iso = IsolationForest(n_estimators=200, random_state=42).fit(X_fit)
score_val_iso = -iso.score_samples(X_val)      # plus c'est grand, plus c'est anormal
score_test_iso = -iso.score_samples(X_test)
evaluer("Isolation Forest", score_test_iso, choisir_seuil(score_val_iso))

# ------------------------------------------------------- 2) autoencodeur
n = X_fit.shape[1]   # 15 caractéristiques
ae = keras.Sequential([
    keras.layers.Input(shape=(n,)),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(4, activation="relu"),      # le « goulot » : 4 chiffres
    keras.layers.Dense(8, activation="relu"),
    keras.layers.Dense(12, activation="relu"),
    keras.layers.Dense(n, activation="linear"),    # reconstruit les 15 valeurs
])
ae.compile(optimizer="adam", loss="mse")

arret = keras.callbacks.EarlyStopping(monitor="val_loss", patience=8,
                                      restore_best_weights=True)
ae.fit(X_fit, X_fit, validation_data=(X_val, X_val),
       epochs=100, batch_size=32, callbacks=[arret], verbose=0)


def erreur(X):
    """Erreur de reconstruction de chaque fenêtre (moyenne des carrés)."""
    return np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)


err_val, err_test = erreur(X_val), erreur(X_test)
seuil = choisir_seuil(err_val)
evaluer("Autoencodeur", err_test, seuil)

# ------------------------------------------------------------ sauvegardes
ae.save(MODEL / "autoencoder.keras")
with open(MODEL / "threshold.json", "w") as f:
    json.dump({"seuil": seuil}, f)

plt.figure(figsize=(12, 4))
plt.semilogy(err_test, color="#0f4c3a", linewidth=1, label="erreur de reconstruction")
plt.fill_between(range(len(y_test)), err_test.min(), err_test.max(),
                 where=y_test == 1, color="red", alpha=0.2, label="anomalie réelle")
plt.axhline(seuil, color="red", linestyle="--", label="seuil")
plt.xlabel("numéro de fenêtre (une toutes les 0,5 s)")
plt.ylabel("erreur (échelle log)")
plt.legend()
plt.tight_layout()
plt.savefig(MODEL / "erreurs_test.png", dpi=120)
print("\nModèle, seuil et graphique sauvegardés dans model/")
