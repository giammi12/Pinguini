"""Servizio web per prevedere la specie di un pinguino con il modello k-NN."""
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]

# Intervalli accettati (controllo di sicurezza sui dati inseriti)
LIMITI = {
    "bill_length_mm": (20, 70),
    "bill_depth_mm": (8, 28),
    "flipper_length_mm": (150, 260),
    "body_mass_g": (2000, 7500),
}

app = Flask(__name__)

# Il modello viene caricato UNA sola volta, all'avvio del servizio
modello = joblib.load("modello_knn.joblib")


def leggi_misure(dati):
    """Converte i dati ricevuti in numeri e controlla che siano plausibili.
    Restituisce (valori, errore): se errore non è None i dati non sono validi."""
    valori = {}
    for nome in FEATURES:
        try:
            v = float(str(dati.get(nome, "")).replace(",", "."))
        except ValueError:
            return None, f"Valore non valido per '{nome}'."
        minimo, massimo = LIMITI[nome]
        if not (minimo <= v <= massimo):
            return None, f"'{nome}' deve essere tra {minimo} e {massimo}."
        valori[nome] = v
    return valori, None


def prevedi(valori):
    riga = pd.DataFrame([valori], columns=FEATURES)
    specie = modello.predict(riga)[0]
    probabilita = dict(zip(modello.classes_, modello.predict_proba(riga)[0]))
    return specie, {k: round(float(p), 3) for k, p in probabilita.items()}


@app.route("/", methods=["GET", "POST"])
def home():
    risultato = errore = None
    valori_inseriti = {}
    if request.method == "POST":
        valori_inseriti = request.form.to_dict()
        valori, errore = leggi_misure(request.form)
        if errore is None:
            specie, prob = prevedi(valori)
            risultato = {"specie": specie, "probabilita": prob}
    return render_template("index.html", risultato=risultato, errore=errore,
                           valori=valori_inseriti, limiti=LIMITI)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    dati = request.get_json(force=True, silent=True) or {}
    valori, errore = leggi_misure(dati)
    if errore:
        return jsonify({"errore": errore}), 400
    specie, prob = prevedi(valori)
    return jsonify({"specie": specie, "probabilita": prob})


@app.route("/health")
def health():
    return jsonify({"stato": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
