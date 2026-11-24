"""Écoute le sujet machine/alerte et affiche chaque message reçu.

À lancer dans un terminal séparé (utile pour tester sans le dashboard).
"""
import json

import paho.mqtt.client as mqtt

TOPIC = "machine/alerte"


def on_connect(client, userdata, *args):
    client.subscribe(TOPIC)
    print("En écoute sur", TOPIC, "(Ctrl+C pour arrêter)")


def on_message(client, userdata, msg):
    m = json.loads(msg.payload)
    drapeau = "ALERTE " if m["etat"] == "alerte" else "normal "
    print(f"[{m['temps_s']:6.1f} s] {drapeau} erreur={m['erreur']:<9} "
          f"(seuil {m['seuil']})")


try:
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)   # paho-mqtt 2.x
except AttributeError:
    client = mqtt.Client()                                   # paho-mqtt 1.x
client.on_connect = on_connect
client.on_message = on_message
client.connect("localhost", 1883)
client.loop_forever()
