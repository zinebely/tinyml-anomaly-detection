"""Étape 7 : dashboard Streamlit qui s'abonne à MQTT.

Lancer :  streamlit run dashboard.py
Prérequis : Mosquitto démarré, puis python simulate_board.py dans un autre terminal.
"""
import json
import threading
from collections import deque

import altair as alt
import pandas as pd
import paho.mqtt.client as mqtt
import streamlit as st

HOST, PORT, TOPIC = "localhost", 1883, "machine/alerte"
MAX_MESURES = 600      # 6 s de signal affichées (100 mesures/s)
MAX_FENETRES = 240     # 2 minutes d'historique (une fenêtre toutes les 0,5 s)

st.set_page_config(page_title="Détection d'anomalies", layout="wide")


class Magasin:
    """Données reçues par MQTT, partagées entre le thread MQTT et Streamlit."""

    def __init__(self):
        self.lock = threading.Lock()
        self.erreur_connexion = None
        self.vider()

    def vider(self):
        with self.lock:
            self.fenetres = deque(maxlen=MAX_FENETRES)
            self.signal = deque(maxlen=MAX_MESURES)


@st.cache_resource
def demarrer_mqtt():
    """Crée le client MQTT une seule fois (Streamlit relance le script souvent)."""
    magasin = Magasin()

    def on_connect(client, userdata, *args):
        client.subscribe(TOPIC)

    def on_message(client, userdata, msg):
        try:
            m = json.loads(msg.payload)
        except ValueError:
            return
        with magasin.lock:
            magasin.fenetres.append(m)
            magasin.signal.extend(m.get("x", []))

    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho-mqtt 2.x
    except AttributeError:
        client = mqtt.Client()                                   # paho-mqtt 1.x
    client.on_connect = on_connect
    client.on_message = on_message
    try:
        client.connect(HOST, PORT)
        client.loop_start()
    except OSError:
        magasin.erreur_connexion = f"Impossible de joindre Mosquitto sur {HOST}:{PORT}."
    return magasin


magasin = demarrer_mqtt()

st.title("Détection d'anomalies sur des vibrations")
st.caption("Version 1 en simulation sur PC · sujet MQTT : " + TOPIC)

with st.sidebar:
    st.header("Contrôles")
    if st.button("Effacer l'historique"):
        magasin.vider()
    st.write("Lancez ensuite `python simulate_board.py` pour envoyer des données.")

if magasin.erreur_connexion:
    st.error(magasin.erreur_connexion + " Démarrez le broker puis rechargez la page.")
    st.stop()


@st.fragment(run_every=1)
def afficher():
    with magasin.lock:
        fenetres = list(magasin.fenetres)
        signal = list(magasin.signal)

    if not fenetres:
        st.info("En attente de messages… lancez `python simulate_board.py`.")
        return

    df = pd.DataFrame(fenetres)
    dernier = fenetres[-1]
    est_alerte = df["etat"] == "alerte"

    # Un épisode = une suite de fenêtres consécutives en alerte (= une panne)
    df["episode"] = (est_alerte != est_alerte.shift()).cumsum().where(est_alerte)
    episodes = []
    for _, g in df[est_alerte].groupby("episode"):
        ep = {
            "début (s)": g["temps_s"].iloc[0],
            "fin (s)": g["temps_s"].iloc[-1],
            "fenêtres en alerte": len(g),
            "erreur max": round(g["erreur"].max(), 1),
        }
        if "anomalie_reelle" in g:
            ep["vraie panne"] = "oui" if g["anomalie_reelle"].max() else "non (fausse alerte)"
        episodes.append(ep)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("État actuel", "ALERTE" if dernier["etat"] == "alerte" else "Normal")
    c2.metric("Erreur / seuil", f"{dernier['erreur']:.2f}", f"seuil {dernier['seuil']:.2f}",
              delta_color="off")
    c3.metric("Fenêtres analysées", len(df))
    c4.metric("Pannes détectées", len(episodes),
              f"{int(est_alerte.sum())} fenêtres en alerte", delta_color="off")

    if dernier["etat"] == "alerte":
        st.error(f"Anomalie détectée à {dernier['temps_s']:.1f} s")

    st.subheader("Signal du capteur (axe x, 6 dernières secondes)")
    if signal:
        t_fin = dernier["temps_s"]
        temps = [round(t_fin - (len(signal) - 1 - k) / 100, 2) for k in range(len(signal))]
        sig = pd.DataFrame({"temps (s)": temps, "axe x (m/s²)": signal}).set_index("temps (s)")
        st.line_chart(sig, height=220)
    else:
        st.caption("Aucune mesure brute reçue : mettez à jour simulate_board.py.")

    st.subheader("Erreur du modèle et seuil")
    tr = df[["temps_s", "erreur", "seuil"]].copy()
    tr["erreur"] = tr["erreur"].clip(lower=1e-4)     # l'échelle log n'accepte pas 0
    base = alt.Chart(tr).encode(x=alt.X("temps_s:Q", title="temps (s)"))
    log = alt.Scale(type="log")
    courbe = base.mark_line(color="#0f4c3a").encode(
        y=alt.Y("erreur:Q", scale=log, title="erreur (échelle log)"))
    ligne_seuil = base.mark_line(color="red", strokeDash=[5, 4]).encode(
        y=alt.Y("seuil:Q", scale=log))
    st.altair_chart((courbe + ligne_seuil).properties(height=240), use_container_width=True)
    st.caption("Ligne rouge pointillée : seuil d'alerte. Échelle logarithmique : "
               "les pannes très fortes n'écrasent plus le reste de la courbe.")

    st.subheader("Pannes détectées")
    if not episodes:
        st.caption("Aucune alerte pour l'instant.")
    else:
        st.dataframe(pd.DataFrame(episodes).iloc[::-1], hide_index=True,
                     use_container_width=True)
        st.caption("Une panne de 5 s couvre environ 10 fenêtres de 1 s (une toutes les 0,5 s) : "
                   "chaque fenêtre en alerte compte pour une alerte, mais elles forment une seule panne.")


afficher()
