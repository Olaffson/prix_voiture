"""
Statistiques de référence pour la détection du drift.

La référence décrit les données sur lesquelles le modèle en service a été
entraîné (répartition de chaque colonne), sa performance sur le jeu de test
et l'importance de chaque colonne pour le modèle. Les nouveaux lots de
données sont comparés à cette référence.

Utilisation :
    python -m monitoring.reference [données_nettoyées.csv] [modèle.pkl] [reference.json]
    (par défaut : data/data_utilisable.csv, streamlit/model.pkl et monitoring/reference.json)
"""
import datetime
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipeline.entrainement import CIBLE, DONNEES_PAR_DEFAUT, MODELE_PAR_DEFAUT, RANDOM_STATE, evaluer

REFERENCE_PAR_DEFAUT = Path(__file__).parent / 'reference.json'

# Nombre d'intervalles pour découper les colonnes numériques. Il reste faible
# car les lots sont petits : avec plus d'intervalles, chacun contiendrait trop
# peu de véhicules et le PSI varierait fortement par simple hasard.
NB_INTERVALLES = 5

# Les valeurs texte plus rares que cette part sont regroupées dans « autres »,
# pour la même raison
PART_MINIMALE_CATEGORIE = 0.05
AUTRES = '__autres__'


def repartition_numerique(serie: pd.Series) -> dict:
    """Découpe une colonne numérique en intervalles de même effectif et renvoie leurs bornes et proportions."""
    bornes = np.unique(np.quantile(serie.dropna(), np.linspace(0, 1, NB_INTERVALLES + 1))[1:-1])
    return {'bornes': bornes.tolist(), 'proportions': proportions_numeriques(serie, bornes.tolist())}


def proportions_numeriques(serie: pd.Series, bornes: list) -> list:
    """Part des valeurs dans chaque intervalle délimité par les bornes (les extrêmes sont ouverts)."""
    indices = np.searchsorted(bornes, serie.dropna(), side='right')
    comptes = np.bincount(indices, minlength=len(bornes) + 1)
    return (comptes / comptes.sum()).tolist()


def repartition_texte(serie: pd.Series) -> dict:
    """Proportion de chaque valeur d'une colonne texte, les valeurs rares étant regroupées."""
    parts = serie.value_counts(normalize=True)
    frequentes = parts[parts >= PART_MINIMALE_CATEGORIE].index.tolist()
    return {'categories': frequentes, 'proportions': proportions_texte(serie, frequentes)}


def proportions_texte(serie: pd.Series, categories: list) -> dict:
    """Part de chaque catégorie, les autres valeurs (rares ou inconnues) étant regroupées dans « autres »."""
    regroupee = serie.where(serie.isin(categories), AUTRES)
    parts = regroupee.value_counts(normalize=True)
    return {categorie: float(parts.get(categorie, 0)) for categorie in [*categories, AUTRES]}


def importances_par_colonne(model: Pipeline) -> dict:
    """
    Importance de chaque colonne d'origine pour la forêt aléatoire.

    L'encodage one-hot transforme une colonne texte en plusieurs colonnes : leurs
    importances sont additionnées pour revenir à la colonne d'origine.
    """
    preprocesseur, regresseur = model[0], model[-1]
    importances = iter(regresseur.feature_importances_)
    resultat = {}
    for nom, transformeur, colonnes in preprocesseur.transformers_:
        if nom == 'remainder':
            continue
        encodeur = transformeur[-1]
        if hasattr(encodeur, 'categories_'):
            for colonne, categories in zip(colonnes, encodeur.categories_):
                resultat[colonne] = sum(next(importances) for _ in categories)
        else:
            for colonne in colonnes:
                resultat[colonne] = next(importances)
    return {colonne: float(valeur) for colonne, valeur in resultat.items()}


def valeurs_connues(model: Pipeline) -> dict:
    """Valeurs de chaque colonne texte vues par le modèle pendant son entraînement."""
    resultat = {}
    for nom, transformeur, colonnes in model[0].transformers_:
        encodeur = transformeur[-1] if nom != 'remainder' else None
        if hasattr(encodeur, 'categories_'):
            for colonne, categories in zip(colonnes, encodeur.categories_):
                resultat[colonne] = [valeur for valeur in categories.tolist() if not pd.isna(valeur)]
    return resultat


def construire_reference(df: pd.DataFrame, model: Pipeline) -> dict:
    """
    Construit la référence à partir des données d'entraînement du modèle.

    Args:
        df (pd.DataFrame): données nettoyées sur lesquelles le modèle a été entraîné et évalué.
        model (Pipeline): modèle en service.

    Returns:
        dict: référence, enregistrable en JSON.
    """
    X = df.drop(CIBLE, axis=1)
    y = df[CIBLE]
    # même découpage que pipeline.entrainement : la performance de référence est celle du jeu de test
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    colonnes_numeriques = df.select_dtypes(include=np.number).columns
    colonnes_texte = df.columns.difference(colonnes_numeriques, sort=False)
    texte = {colonne: repartition_texte(df[colonne].dropna()) for colonne in colonnes_texte}

    return {
        'date': datetime.date.today().isoformat(),
        'version_scikit_learn': sklearn.__version__,
        'nb_vehicules': len(df),
        'performance': evaluer(model, X_test, y_test),
        'importances': importances_par_colonne(model),
        'valeurs_connues': valeurs_connues(model),
        'numeriques': {colonne: repartition_numerique(df[colonne]) for colonne in colonnes_numeriques},
        # une colonne dont toutes les valeurs sont rares (modele) n'a qu'une catégorie « autres » :
        # son PSI serait toujours nul, elle n'est donc pas suivie
        'texte': {colonne: repartition for colonne, repartition in texte.items() if repartition['categories']},
    }


def charger_reference(chemin: Path = REFERENCE_PAR_DEFAUT) -> dict:
    with open(chemin, encoding='utf-8') as file:
        return json.load(file)


def sauvegarder_reference(reference: dict, chemin: Path = REFERENCE_PAR_DEFAUT) -> None:
    with open(chemin, 'w', encoding='utf-8') as file:
        json.dump(reference, file, ensure_ascii=False, indent=2)
        file.write('\n')


if __name__ == '__main__':
    donnees = Path(sys.argv[1]) if len(sys.argv) > 1 else DONNEES_PAR_DEFAUT
    modele = Path(sys.argv[2]) if len(sys.argv) > 2 else MODELE_PAR_DEFAUT
    destination = Path(sys.argv[3]) if len(sys.argv) > 3 else REFERENCE_PAR_DEFAUT

    with open(modele, 'rb') as file:
        model = pickle.load(file)
    reference = construire_reference(pd.read_csv(donnees), model)
    sauvegarder_reference(reference, destination)

    performance = reference['performance']
    print(f"Référence enregistrée dans {destination} : {reference['nb_vehicules']} véhicules, "
          f"R² {performance['r2']:.3f}, MAE {performance['mae']:.0f} $")
