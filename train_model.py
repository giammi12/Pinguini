"""Addestra il modello k-NN (lo stesso del notebook) e lo salva in modello_knn.joblib.

Uso:  python train_model.py
Il file salvato viene caricato da app.py.
"""
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
K = 5  # il valore di k scelto nel notebook

df = pd.read_csv("data/penguins_clean.csv")
X, y = df[FEATURES], df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

modello = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
modello.fit(X_train, y_train)
print("Accuracy sul test set:", round(modello.score(X_test, y_test), 3))

joblib.dump(modello, "modello_knn.joblib")
print("Modello salvato in modello_knn.joblib")
