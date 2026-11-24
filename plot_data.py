"""Trace le signal de test : les zones anormales sont colorées en rouge."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RACINE = Path(__file__).resolve().parent

df = pd.read_csv(RACINE / "data" / "test_mixed.csv")

fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)

for ax, axe in zip(axes, ["x", "y", "z"]):
    ax.plot(df["t"], df[axe], color="#0f4c3a", linewidth=0.8)
    # fond rouge là où label == 1
    anormal = df["label"] == 1
    ax.fill_between(df["t"], df[axe].min(), df[axe].max(),
                    where=anormal, color="red", alpha=0.2, label="anomalie")
    ax.set_ylabel(f"axe {axe} (m/s²)")

axes[0].legend(loc="upper right")
axes[-1].set_xlabel("temps (s)")
fig.suptitle("Signal simulé du capteur : zones anormales en rouge")
plt.tight_layout()
plt.savefig(RACINE / "signal_test.png", dpi=120)
print("Graphique enregistré : signal_test.png")
