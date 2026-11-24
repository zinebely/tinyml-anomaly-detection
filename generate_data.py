"""Étape 2 : simuler un accéléromètre (comme le MPU6050) sans capteur.

Produit deux fichiers dans data/ :
- normal_train.csv : 10 minutes de fonctionnement normal (pour entraîner)
- test_mixed.csv   : 2 minutes avec des anomalies (pour tester)
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent / "data"

FS = 100            # fréquence d'échantillonnage : 100 mesures par seconde
G = 9.81            # gravité : l'axe z « sent » toujours environ 9.81
rng = np.random.default_rng(42)   # graine fixe : mêmes résultats à chaque lancement


def signal_normal(n, freq=5.0, amp=0.05, bruit=0.02):
    """n mesures de vibration normale : une petite onde régulière + du bruit."""
    t = np.arange(n) / FS
    x = amp * np.sin(2 * np.pi * freq * t) + rng.normal(0, bruit, n)
    y = amp * np.sin(2 * np.pi * freq * t + 1.0) + rng.normal(0, bruit, n)
    z = G + amp * np.sin(2 * np.pi * freq * t + 2.0) + rng.normal(0, bruit, n)
    return t, np.column_stack([x, y, z])


def to_df(t, xyz, label):
    return pd.DataFrame({"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2],
                         "label": label})


def zone(debut_s, fin_s):
    return slice(int(debut_s * FS), int(fin_s * FS))


def main():
    DATA.mkdir(exist_ok=True)

    # 1) Données d'entraînement : uniquement du normal (600 s)
    t, xyz = signal_normal(600 * FS)
    to_df(t, xyz, np.zeros(len(t), dtype=int)).to_csv(DATA / "normal_train.csv", index=False)

    # 2) Données de test : 120 s de normal, avec 3 types d'anomalies insérés
    t, xyz = signal_normal(120 * FS)
    label = np.zeros(len(t), dtype=int)   # 0 = normal, 1 = anormal

    # Anomalie A : vibration forte (amplitude x8) entre 20 s et 25 s
    z_a = zone(20, 25)
    _, fort = signal_normal(len(xyz[z_a]), amp=0.4, bruit=0.15)
    xyz[z_a] = fort
    label[z_a] = 1

    # Anomalie B : changement de fréquence (5 Hz -> 18 Hz) entre 50 s et 55 s
    z_b = zone(50, 55)
    _, autre = signal_normal(len(xyz[z_b]), freq=18.0)
    xyz[z_b] = autre
    label[z_b] = 1

    # Anomalie C : chocs ponctuels (pics brusques) entre 80 s et 85 s
    z_c = zone(80, 85)
    for i in range(z_c.start, z_c.stop, 40):          # un choc toutes les 0,4 s
        xyz[i:i + 3] += rng.normal(0, 2.5, (3, 3))
    label[z_c] = 1

    to_df(t, xyz, label).to_csv(DATA / "test_mixed.csv", index=False)
    print("Fichiers créés dans data/ :")
    print(" - normal_train.csv :", 600 * FS, "lignes")
    print(" - test_mixed.csv   :", len(t), "lignes dont", int(label.sum()), "anormales")


if __name__ == "__main__":
    main()
