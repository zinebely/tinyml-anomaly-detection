"""Calcul des caractéristiques : LE SEUL endroit où ce calcul est écrit.

Ce fichier est importé par extract_features.py (entraînement) et par
simulate_board.py (simulation de la carte). Ainsi, le modèle reçoit toujours
des caractéristiques calculées exactement de la même façon.

Sur l'ESP32 (version 2), ce calcul devra être réécrit en C++ à l'identique.
"""
import numpy as np

FS = 100          # mesures par seconde
FENETRE = 100     # 1 seconde = 100 mesures
PAS = 50          # on avance de 0,5 s : recouvrement de 50 %
BANDES = [(0, 10), (10, 25), (25, 50)]   # bandes de fréquence en Hz

# Noms des 15 colonnes, dans l'ordre produit par caracteristiques_fenetre()
NOMS = [f"{axe}_{c}" for axe in "xyz"
        for c in ["rms", "pic", "bande_0_10", "bande_10_25", "bande_25_50"]]


def caracteristiques_axe(v):
    """Résume un morceau de signal d'un axe en 5 chiffres."""
    v = v - v.mean()                                # enlève la gravité (z ≈ 9,81)
    rms = np.sqrt(np.mean(v ** 2))                  # force moyenne des vibrations
    pic = np.max(np.abs(v))                         # plus grand pic
    spectre = np.abs(np.fft.rfft(v)) ** 2           # énergie par fréquence (FFT)
    freqs = np.fft.rfftfreq(len(v), d=1 / FS)
    bandes = [spectre[(freqs >= a) & (freqs < b)].sum() for a, b in BANDES]
    return [rms, pic] + bandes


def caracteristiques_fenetre(fenetre_xyz):
    """Une fenêtre brute (FENETRE x 3 : colonnes x, y, z) -> 15 caractéristiques."""
    ligne = []
    for k in range(3):
        ligne += caracteristiques_axe(fenetre_xyz[:, k])
    return np.array(ligne, dtype="float64")


def normaliser(caracteristiques, moyenne, ecart):
    """(valeur - moyenne) / écart-type, avec les valeurs calculées sur le train."""
    return (np.asarray(caracteristiques) - moyenne) / ecart
