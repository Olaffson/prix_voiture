# prix_voiture

Application d'estimation du prix d'un véhicule, construite avec Python, scikit-learn et Streamlit.

Les données fournies (`data/carprice.csv`) sont nettoyées, puis utilisées pour entraîner un modèle de régression (forêt aléatoire) avec scikit-learn. Le modèle est enregistré avec pickle dans `streamlit/model.pkl` et chargé par l'application Streamlit `streamlit/app.py`.

Les prix sont exprimés en **dollars américains**, la devise des données d'origine.

## Structure du projet

| Dossier / fichier | Contenu |
|---|---|
| `data/carprice.csv` | Données brutes (noms de colonnes en anglais, unités américaines) |
| `data/data_utilisable.csv` | Données nettoyées : colonnes en français, unités métriques, marque et modèle séparés |
| `data/donnees_modele.csv` | Données d'entraînement du modèle en service, utilisées pour les listes de l'application |
| `notebook/nettoyage.ipynb` | Exploration et nettoyage des données |
| `notebook/model.ipynb` | Exploration du modèle : évaluation, courbe d'apprentissage, recherche d'hyperparamètres |
| `pipeline/nettoyage.py` | Nettoyage des données brutes, réutilisable pour de nouvelles données |
| `pipeline/entrainement.py` | Entraînement du modèle et enregistrement dans `streamlit/model.pkl` |
| `streamlit/app.py` | Interface de l'application |
| `streamlit/components.py` | Fonction de prédiction du prix |
| `streamlit/model.pkl` | Modèle entraîné utilisé par l'application |
| `simulation/generer_lot.py` | Génération de lots de données simulés, pour tester le suivi du drift |
| `simulation/scenario.toml` | Scénario de drift appliqué aux lots simulés |
| `data/nouvelles/` | Lots de données simulés |
| `monitoring/reference.py` | Statistiques de référence du modèle en service (`monitoring/reference.json`) |
| `monitoring/drift.py` | Détection du drift d'un lot et décision de réentraînement |
| `monitoring/reentrainement.py` | Réentraînement automatique et remplacement du modèle en service |
| `monitoring/seuils.toml` | Seuils de décision et réglages du réentraînement |
| `monitoring/rapports/` | Rapports de drift et de réentraînement |
| `modeles/archives/` | Anciens modèles, avec leur référence et leurs données |
| `tests/` | Tests du nettoyage, de l'entraînement, de la simulation, du suivi du drift et de l'application |

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

## Repartir du modèle d'origine

Pour réentraîner le modèle sur les seules données d'origine, par exemple pour revenir au point de départ, lancez depuis la racine du dépôt :

```bash
# nettoyage des données brutes
python -m pipeline.nettoyage data/carprice.csv data/data_utilisable.csv

# entraînement du modèle, enregistré dans streamlit/model.pkl
python -m pipeline.entrainement

# données et référence du modèle en service
cp data/data_utilisable.csv data/donnees_modele.csv
python -m monitoring.reference
```

L'entraînement affiche les scores du modèle (R², RMSE et MAE) sur les jeux d'entraînement et de test. Le nouveau modèle est aussitôt utilisé par l'application.

Les trois dernières commandes vont ensemble : le modèle, ses données et sa référence doivent toujours correspondre.

Un modèle enregistré avec pickle ne se recharge de façon fiable qu'avec la version de scikit-learn qui l'a entraîné : si vous changez cette version, mettez à jour `requirements.txt` et réentraînez le modèle.

## Simulation de nouvelles données

Faute de nouvelles données réelles, des lots sont simulés pour tester la détection du drift et le réentraînement du modèle :

```bash
python -m simulation.generer_lot
```

Chaque appel crée le lot suivant dans `data/nouvelles/lot_NNN_AAAA-MM-JJ.csv` (60 véhicules, au format de `data/carprice.csv`). Les véhicules sont tirés des données d'origine, légèrement modifiés, puis déformés selon le scénario de `simulation/scenario.toml` :

| Drift simulé | Réglage par défaut |
|---|---|
| Inflation des prix | +1,5 % par lot |
| Moteurs plus puissants | +1 % de puissance par lot |
| Changement de gamme | diesel, turbo et berlines de plus en plus fréquents |
| Poids croissant de la puissance dans le prix | le prix dépend de plus en plus de la puissance |
| Nouvelles marques | tesla à partir du lot 4, kia à partir du lot 6 |

Le drift augmente avec le numéro du lot. Avec ces réglages, le modèle actuel reste fiable sur les lots 1 et 2, est à la limite au lot 3, puis se dégrade nettement à partir du lot 4.

Un lot est reproductible : `python -m simulation.generer_lot --numero 3` régénère toujours le même lot 3.

Ces données sont fictives : elles servent à tester la chaîne de suivi du modèle, pas à l'améliorer.

## Détection du drift

Chaque nouveau lot est comparé à la référence, c'est-à-dire aux données d'entraînement et à la performance du modèle en service :

```bash
python -m monitoring.drift data/nouvelles/lot_001_2026-10-05.csv
```

L'analyse porte sur trois points :

- **la performance du modèle sur le lot**, grâce aux prix réels : erreur moyenne (MAE) et R² ;
- **le drift de chaque colonne**, mesuré par le PSI (*Population Stability Index*). Un drift n'est retenu que s'il est confirmé par un test du khi-deux : sur un petit lot, le PSI varie beaucoup par simple hasard ;
- **les valeurs inconnues du modèle**, comme une nouvelle marque, que le modèle ignore sans signaler d'erreur.

| Décision | Condition (seuils de `monitoring/seuils.toml`) |
|---|---|
| **réentraîner** | MAE > 1,2 × la MAE de référence, ou R² < 0,80, ou colonnes en drift important représentant au moins 30 % de l'importance du modèle, ou plus de 5 % de véhicules avec une valeur inconnue |
| **surveiller** | drift modéré d'une colonne, ou quelques valeurs inconnues |
| **aucune action** | rien de notable |
| **lot trop petit** | moins de 50 véhicules : les mesures seraient trop instables |

Un rapport Markdown et un résultat JSON sont enregistrés dans `monitoring/rapports/`.

Réglés sur des lots simulés, ces seuils déclenchent un réentraînement dans 2 % des cas sur des lots sans drift, et dans 100 % des cas à partir du lot 4 du scénario.

## Réentraînement automatique

Quand l'analyse d'un lot conclut qu'il faut réentraîner le modèle :

```bash
python -m monitoring.reentrainement
```

1. **Données d'entraînement** : les lots les plus récents de `data/nouvelles/`, jusqu'à atteindre au moins 200 véhicules (fenêtre glissante), complétés si besoin par des véhicules des données d'origine. Avec l'inflation et l'évolution de la gamme, les données anciennes ne sont plus représentatives.
2. **Comparaison** : un nouveau modèle est évalué par validation croisée sur ces données. Sur le dernier lot, ses prédictions sont comparées à celles du modèle en service, qui n'a jamais vu ce lot.
3. **Mise en service**, seulement si le nouveau modèle a une erreur moyenne plus faible sur le dernier lot :
   - l'ancien modèle, sa référence et ses données sont archivés dans `modeles/archives/<lot>/` ;
   - le nouveau modèle est entraîné sur toutes les données de la fenêtre et remplace `streamlit/model.pkl` ;
   - `data/donnees_modele.csv` et `monitoring/reference.json` sont mis à jour : les listes de l'application proposent les nouvelles valeurs, comme une nouvelle marque, et les prochains lots sont comparés au nouveau modèle.

Un rapport est enregistré dans `monitoring/rapports/reentrainement_<lot>.md`.

Pour revenir à un modèle archivé, copiez les trois fichiers de son dossier d'archive à leur place (`streamlit/`, `monitoring/`, `data/`).

Sur 16 semaines simulées, le cycle a mis en service 8 nouveaux modèles et en a écarté 3 qui n'étaient pas meilleurs. Au lot 16, l'erreur moyenne du modèle en service est de 4 637 $, contre 12 232 $ pour le modèle d'origine jamais réentraîné.

## Tests

```bash
pytest
```

## Licence

Ce projet est sous licence MIT, voir [LICENSE](LICENSE) pour plus de détails.

## Auteur

* **Olivier Kotwica** [@Olaffson](https://github.com/Olaffson)
