"""
Entraînement du modèle d'estimation du prix.

Reprend le pipeline de notebook/model.ipynb : standardisation des colonnes
numériques, encodage one-hot des colonnes texte, puis forêt aléatoire.

Utilisation :
    python -m pipeline.entrainement [données_nettoyées.csv] [modèle.pkl]
    (par défaut : data/data_utilisable.csv et streamlit/model.pkl)
"""
import pickle
import sys
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RACINE = Path(__file__).parent.parent
DONNEES_PAR_DEFAUT = RACINE / 'data' / 'data_utilisable.csv'
MODELE_PAR_DEFAUT = RACINE / 'streamlit' / 'model.pkl'

CIBLE = 'prix'
RANDOM_STATE = 42


def creer_modele() -> Pipeline:
    """Crée le pipeline de prétraitement et de régression (non entraîné)."""
    numerical_pipeline = make_pipeline(SimpleImputer(), StandardScaler())
    categorical_pipeline = make_pipeline(OneHotEncoder(handle_unknown='ignore'))

    preprocessor = make_column_transformer(
        (numerical_pipeline, make_column_selector(dtype_include=np.number)),
        (categorical_pipeline, make_column_selector(dtype_exclude=np.number)),
    )
    return make_pipeline(preprocessor, RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE))


def evaluer(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Calcule le R², le RMSE et le MAE du modèle sur les données fournies."""
    return mesurer(y, model.predict(X))


def mesurer(y, y_pred) -> dict:
    """Calcule le R², le RMSE et le MAE de prédictions déjà faites."""
    return {
        'r2': r2_score(y, y_pred),
        'rmse': sqrt(mean_squared_error(y, y_pred)),
        'mae': mean_absolute_error(y, y_pred),
    }


def entrainer(df: pd.DataFrame, test_size: float = 0.2) -> tuple[Pipeline, dict]:
    """
    Entraîne le modèle sur une partie des données et l'évalue sur le reste.

    Args:
        df (pd.DataFrame): données nettoyées, avec la colonne prix.
        test_size (float): part des données gardée pour l'évaluation.

    Returns:
        tuple: le modèle entraîné et ses scores sur les jeux d'entraînement et de test.
    """
    X = df.drop(CIBLE, axis=1)
    y = df[CIBLE]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)

    model = creer_modele()
    model.fit(X_train, y_train)

    scores = {'train': evaluer(model, X_train, y_train), 'test': evaluer(model, X_test, y_test)}
    return model, scores


def sauvegarder(model: Pipeline, chemin: Path) -> None:
    """Enregistre le modèle avec pickle."""
    with open(chemin, 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    donnees = Path(sys.argv[1]) if len(sys.argv) > 1 else DONNEES_PAR_DEFAUT
    destination = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELE_PAR_DEFAUT

    model, scores = entrainer(pd.read_csv(donnees))
    sauvegarder(model, destination)

    for jeu, valeurs in scores.items():
        print(f"{jeu:5} : R² {valeurs['r2']:.3f} | RMSE {valeurs['rmse']:.0f} $ | MAE {valeurs['mae']:.0f} $")
    print(f'Modèle enregistré dans {destination}')
