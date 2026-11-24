"""Étape 6 : simuler la carte ESP32 sur PC.

Lit data/test_mixed.csv ligne par ligne comme le ferait le capteur, puis pour
chaque fenêtre : caractéristiques -> normalisation -> modèle int8 -> alerte MQTT.
Le message contient aussi les 50 dernières mesures de l'axe x ("x") pour le dashboard.

Utilisation :
    python simulate_board.py                 # 10x plus vite que le temps réel
    python simulate_board.py --vitesse 1     # temps réel (100 mesures/s)
    python simulate_board.py --sans-mqtt     # affichage seul, sans Mosquitto
"""
import argparse
import json
import os
import time
from pathlib import Path
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd
import tensorflow as tf
import paho.mqtt.client as mqtt

from features import FS, FENETRE, PAS, caracteristiques_fenetre, normaliser

RACINE = Path(__file__).resolve().parent
DATA, MODEL = RACINE / "data", RACINE / "model"
TOPIC = "machine/alerte"

parser = argparse.ArgumentParser()
parser.add_argument("--vitesse", type=float, default=10.0,
                    help="1 = temps réel, 10 = dix fois plus vite")
parser.add_argument("--sans-mqtt", action="store_true")
args, unknown = parser.parse_known_args()

# ----------------------- ce que la carte « connaît » (fichiers de l'entraînement)
scaler = json.load(open(DATA / "scaler.json"))
moyenne = np.array(scaler["moyenne"], dtype="float32")
ecart = np.array(scaler["ecart"], dtype="float32")
seuil = json.load(open(MODEL / "threshold_int8.json"))["seuil"]

# On lit le fichier nous-mêmes : TFLite ne sait pas ouvrir un chemin contenant des
# accents ou une apostrophe sous Windows (ex. « détection d'anomalies »).
modele_octets = (MODEL / "autoencoder_int8.tflite").read_bytes()
interp = tf.lite.Interpreter(model_content=modele_octets)
interp.allocate_tensors()
entree = interp.get_input_details()[0]
sortie = interp.get_output_details()[0]
e_scale, e_zero = entree["quantization"]
s_scale, s_zero = sortie["quantization"]


def analyser(fenetre_xyz):
    """Une fenêtre brute (100 x 3) -> erreur de reconstruction du modèle."""
    f = caracteristiques_fenetre(fenetre_xyz)
    x = normaliser(f, moyenne, ecart).astype("float32")
    q = np.clip(np.round(x / e_scale + e_zero), -128, 127).astype(np.int8)
    interp.set_tensor(entree["index"], q.reshape(1, -1))
    interp.invoke()
    sortie_q = interp.get_tensor(sortie["index"]).astype(np.float32)
    rec = ((sortie_q - s_zero) * s_scale).ravel()
    return float(np.mean((x - rec) ** 2))


# --------------------------------------------------------------------- MQTT
client = None
if not args.sans_mqtt:
    try:
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho-mqtt 2.x
        except AttributeError:
            client = mqtt.Client()                                   # paho-mqtt 1.x
        client.connect("localhost", 1883)
        client.loop_start()
        print("Connecté à Mosquitto (localhost:1883), sujet :", TOPIC)
    except OSError:
        print("Mosquitto introuvable : affichage seul (voir --sans-mqtt).")
        client = None

# ------------------------------------------------------------ « capteur » CSV
df = pd.read_csv(DATA / "test_mixed.csv")
mesures = df[["x", "y", "z"]].to_numpy()
etiquettes = df["label"].to_numpy()

tampon, nb, alertes = [], 0, 0
for i, ligne in enumerate(mesures):
    tampon.append(ligne)                   # une mesure arrive
    time.sleep(1 / FS / args.vitesse)      # on respecte la cadence du capteur
    if len(tampon) >= FENETRE and (len(tampon) - FENETRE) % PAS == 0:
        fen = np.array(tampon[-FENETRE:])
        erreur = analyser(fen)
        etat = "alerte" if erreur > seuil else "normal"
        message = {
            "temps_s": round((i + 1) / FS, 2),
            "erreur": round(erreur, 4),
            "seuil": round(seuil, 4),
            "etat": etat,
            # les 50 nouvelles mesures de l'axe x (pour tracer le signal)
            "x": [round(float(v), 3) for v in fen[-PAS:, 0]],
            # réalité du simulateur : n'existe pas sur la vraie carte
            "anomalie_reelle": int(etiquettes[i + 1 - FENETRE:i + 1].mean() > 0.5),
        }
        nb += 1
        alertes += etat == "alerte"
        print(f"[{message['temps_s']:6.1f} s] erreur {erreur:8.3f} -> {etat.upper()}")
        if client:
            client.publish(TOPIC, json.dumps(message))

print(f"\nTerminé : {nb} fenêtres analysées, {alertes} alertes.")
if client:
    client.loop_stop()
    client.disconnect()
