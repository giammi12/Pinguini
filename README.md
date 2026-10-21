# Pinguini API – previsione della specie con k-NN

Servizio web Flask che usa un modello k-NN (con StandardScaler) addestrato sul dataset Palmer Penguins.

## Struttura
- `app.py` – applicazione Flask (pagina web + API JSON)
- `templates/index.html` – pagina con il modulo
- `modello_knn.joblib` – modello addestrato
- `train_model.py` – riaddestra il modello e rigenera `modello_knn.joblib`
- `data/penguins_clean.csv` – dataset ripulito
- `requirements.txt` – librerie (versioni fissate)
- `render.yaml` – configurazione facoltativa per Render

## Prova in locale
```
pip install -r requirements.txt
python app.py
```
Apri http://127.0.0.1:5000

## Uso dell'API
```
curl -X POST https://NOME-SERVIZIO.onrender.com/api/predict \
  -H "Content-Type: application/json" \
  -d '{"bill_length_mm": 50, "bill_depth_mm": 15, "flipper_length_mm": 220, "body_mass_g": 5000}'
```

## Attenzione alle versioni
Il file `.joblib` va caricato con la **stessa versione di scikit-learn** usata per salvarlo
(qui 1.8.0). Se esporti un modello da Colab, cambia la riga `scikit-learn==...` in
`requirements.txt` con la versione stampata da `sklearn.__version__`, oppure rigenera il
modello con `python train_model.py`.
