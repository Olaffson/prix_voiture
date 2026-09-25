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
| `notebook/model.ipynb` | Entraînement, évaluation et export du modèle |
| `streamlit/app.py` | Interface de l'application |
| `streamlit/components.py` | Fonction de prédiction du prix |
| `streamlit/model.pkl` | Modèle entraîné utilisé par l'application |

## Installation

Le modèle a été entraîné avec scikit-learn 1.2.2, qui ne fonctionne qu'avec **Python 3.12 au maximum**.

1. Clonez ce dépôt : `git clone git@github.com:Olaffson/prix_voiture.git`
2. Créez un environnement virtuel avec Python 3.12 ou une version antérieure, puis activez-le :
   ```bash
   python3.12 -m venv venv
   source venv/bin/activate
   ```
3. Installez les dépendances : `pip install -r requirements.txt`

## Application

Depuis la racine du dépôt, lancez :

```bash
streamlit run streamlit/app.py
```

Sélectionnez les caractéristiques du véhicule, puis cliquez sur **Estimation** pour afficher le prix estimé.

## Réentraîner le modèle

Exécutez `notebook/model.ipynb` : il lit `data/data_utilisable.csv`, entraîne le modèle et l'enregistre dans `streamlit/model.pkl`, qui est aussitôt utilisé par l'application.

Si vous réentraînez le modèle avec une autre version de scikit-learn, mettez à jour la version indiquée dans `requirements.txt`.

## Licence

Ce projet est sous licence MIT, voir [LICENSE](LICENSE) pour plus de détails.

## Auteur

* **Olivier Kotwica** [@Olaffson](https://github.com/Olaffson)
