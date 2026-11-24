"""Étape 5 : alléger le modèle (TensorFlow Lite, quantification 8 bits).

Entrées : model/autoencoder.keras, data/features_train.csv, data/features_test.csv
Sorties : model/autoencoder_float.tflite, model/autoencoder_int8.tflite,
          model/threshold_int8.json
"""
import json
import os
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import precision_score, recall_score, f1_score

RACINE = Path(__file__).resolve().parent
DATA, MODEL = RACINE / "data", RACINE / "model"

# ---------------------------------------------------------------- données
train = pd.read_csv(DATA / "features_train.csv")
test = pd.read_csv(DATA / "features_test.csv")
NOMS = [c for c in train.columns if c != "label"]
X_train = train[NOMS].to_numpy(dtype="float32")
X_test = test[NOMS].to_numpy(dtype="float32")
y_test = test["label"].to_numpy()
coupure = int(0.8 * len(X_train))
X_fit, X_val = X_train[:coupure], X_train[coupure:]   # même découpage que train.py

ae = keras.models.load_model(MODEL / "autoencoder.keras")

# --------------------------------------- 1) conversion sans quantification
conv = tf.lite.TFLiteConverter.from_keras_model(ae)
tflite_float = conv.convert()
(MODEL / "autoencoder_float.tflite").write_bytes(tflite_float)


# ------------------------------------------- 2) conversion en entiers 8 bits
def donnees_exemple():
    """Quelques fenêtres normales pour calibrer l'arrondi."""
    for i in range(0, len(X_fit), 5):
        yield [X_fit[i:i + 1]]


conv = tf.lite.TFLiteConverter.from_keras_model(ae)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = donnees_exemple
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type = tf.int8      # entrée et sortie en entiers, comme sur la carte
conv.inference_output_type = tf.int8
tflite_int8 = conv.convert()
(MODEL / "autoencoder_int8.tflite").write_bytes(tflite_int8)

# ----------------------------------------- 3) exécuter le modèle int8 sur PC
interp = tf.lite.Interpreter(model_content=tflite_int8)
interp.allocate_tensors()
entree = interp.get_input_details()[0]
sortie = interp.get_output_details()[0]
e_scale, e_zero = entree["quantization"]
s_scale, s_zero = sortie["quantization"]


def reconstruire_int8(x):
    """Une fenêtre : quantifier l'entrée, exécuter, déquantifier la sortie."""
    q = np.clip(np.round(x / e_scale + e_zero), -128, 127).astype(np.int8)
    interp.set_tensor(entree["index"], q.reshape(1, -1))
    interp.invoke()
    out = interp.get_tensor(sortie["index"]).astype(np.float32)
    return ((out - s_zero) * s_scale).ravel()


def erreurs_int8(X):
    return np.array([np.mean((x - reconstruire_int8(x)) ** 2) for x in X])


def erreurs_keras(X):
    return np.mean((X - ae.predict(X, verbose=0)) ** 2, axis=1)


err_val_f, err_test_f = erreurs_keras(X_val), erreurs_keras(X_test)
err_val_q, err_test_q = erreurs_int8(X_val), erreurs_int8(X_test)
seuil_f = float(np.percentile(err_val_f, 99))
seuil_q = float(np.percentile(err_val_q, 99))     # seuil recalculé pour le modèle int8
with open(MODEL / "threshold_int8.json", "w") as f:
    json.dump({"seuil": seuil_q}, f)


# ------------------------------------------------------------- comparaison
def resume(nom, err, seuil):
    y = (err > seuil).astype(int)
    print(f"{nom:<22} précision {precision_score(y_test, y, zero_division=0):.2f} | "
          f"rappel {recall_score(y_test, y, zero_division=0):.2f} | "
          f"F1 {f1_score(y_test, y, zero_division=0):.2f}")
    return y


print("=== Tailles ===")
print(f"Keras (.keras)         : {os.path.getsize(MODEL / 'autoencoder.keras') / 1024:.1f} Ko")
print(f"TFLite float32         : {len(tflite_float) / 1024:.1f} Ko")
print(f"TFLite int8            : {len(tflite_int8) / 1024:.1f} Ko")

print("\n=== Détection sur le jeu de test ===")
y_f = resume("Keras float32", err_test_f, seuil_f)
y_q = resume("TFLite int8", err_test_q, seuil_q)
print(f"\nDécisions identiques sur {np.mean(y_f == y_q) * 100:.1f} % des fenêtres")
