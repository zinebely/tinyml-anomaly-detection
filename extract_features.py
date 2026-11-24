"""Étape 3 : transformer le signal brut en un tableau de caractéristiques.

Entrées  : data/normal_train.csv, data/test_mixed.csv
Sorties  : data/features_train.csv, data/features_test.csv, data/scaler.json

Le calcul des caractéristiques lui-même est dans features.py.
"""
import json
from pathlib import Path

import pandas as pd

from features import FENETRE, PAS, NOMS, caracteristiques_fenetre

DATA = Path(__file__).resolve().parent / "data"


def fenetres(df):
    """Découpe un signal en fenêtres glissantes et calcule leurs caractéristiques."""
    lignes, etiquettes = [], []
    for debut in range(0, len(df) - FENETRE + 1, PAS):
        morceau = df.iloc[debut:debut + FENETRE]
        lignes.append(caracteristiques_fenetre(morceau[["x", "y", "z"]].to_numpy()))
        # la fenêtre est « anormale » si plus de la moitié de ses mesures le sont
        etiquettes.append(int(morceau["label"].mean() > 0.5))
    out = pd.DataFrame(lignes, columns=NOMS)
    out["label"] = etiquettes
    return out


def main():
    train = fenetres(pd.read_csv(DATA / "normal_train.csv"))
    test = fenetres(pd.read_csv(DATA / "test_mixed.csv"))

    # Normalisation : on utilise UNIQUEMENT les statistiques du normal (train)
    moy = train[NOMS].mean()
    ecart = train[NOMS].std().replace(0, 1)
    train[NOMS] = (train[NOMS] - moy) / ecart
    test[NOMS] = (test[NOMS] - moy) / ecart

    train.to_csv(DATA / "features_train.csv", index=False)
    test.to_csv(DATA / "features_test.csv", index=False)
    with open(DATA / "scaler.json", "w") as f:
        json.dump({"moyenne": moy.tolist(), "ecart": ecart.tolist(), "noms": NOMS}, f)

    print("Fenêtres train :", len(train), "| test :", len(test),
          "dont anormales :", int(test["label"].sum()))
    print("Nombre de caractéristiques par fenêtre :", len(NOMS))


if __name__ == "__main__":
    main()
