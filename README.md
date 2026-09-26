# prix_voiture

Application d'estimation du prix d'un véhicule, construite avec Python, scikit-learn et Streamlit.

Les données fournies (`data/carprice.csv`) sont nettoyées, puis utilisées pour entraîner un modèle de régression (forêt aléatoire) avec scikit-learn. Le modèle est enregistré avec pickle dans `streamlit/model.pkl` et chargé par l'application Streamlit `streamlit/app.py`.

Les prix sont exprimés en **dollars américains**, la devise des données d'origine.

## Structure du projet

| Dossier / fichier | Contenu |
|---|---|
| `data/carprice.csv` | Données brutes (noms de colonnes en anglais, unités américaines) |
| `data/data_utilisable.csv` | Données nettoyées : colonnes en français, unités métriques, marque et modèle séparés |
| `notebook/nettoyage.ipynb` | Exploration et nettoyage des données |
| `notebook/model.ipynb` | Exploration du modèle : évaluation, courbe d'apprentissage, recherche d'hyperparamètres |
| `pipeline/nettoyage.py` | Nettoyage des données brutes, réutilisable pour de nouvelles données |
| `pipeline/entrainement.py` | Entraînement du modèle et enregistrement dans `streamlit/model.pkl` |
| `streamlit/app.py` | Interface de l'application |
| `streamlit/components.py` | Fonction de prédiction du prix |
| `streamlit/model.pkl` | Modèle entraîné utilisé par l'application |
| `tests/` | Tests du nettoyage et de l'entraînement |

## Installation

Le projet nécessite **Python 3.11, 3.12 ou 3.13**.

1. Clonez ce dépôt : `git clone git@github.com:Olaffson/prix_voiture.git`
2. Créez un environnement virtuel, puis activez-le :
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Installez les dépendances : `pip install -r requirements.txt`

## Application

Depuis la racine du dépôt, lancez :

```bash
streamlit run streamlit/app.py
```

Sélectionnez les caractéristiques du véhicule, puis cliquez sur **Estimation** pour afficher le prix estimé.

## Préparer les données et réentraîner le modèle

Depuis la racine du dépôt :

```bash
# nettoyage des données brutes
python -m pipeline.nettoyage data/carprice.csv data/data_utilisable.csv

# entraînement du modèle, enregistré dans streamlit/model.pkl
python -m pipeline.entrainement
```

L'entraînement affiche les scores du modèle (R², RMSE et MAE) sur les jeux d'entraînement et de test. Le nouveau modèle est aussitôt utilisé par l'application.

Un modèle enregistré avec pickle ne se recharge de façon fiable qu'avec la version de scikit-learn qui l'a entraîné : si vous changez cette version, mettez à jour `requirements.txt` et réentraînez le modèle.

## Tests

```bash
pytest
```

## Licence

Ce projet est sous licence MIT, voir [LICENSE](LICENSE) pour plus de détails.

## Auteur

* **Olivier Kotwica** [@Olaffson](https://github.com/Olaffson)
