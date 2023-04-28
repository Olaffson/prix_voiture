# prix_voiture
Ce projet est une application d'estimation de prix d'un véhicule construite avec Streamlit et Python. Après nettoyage des données fournies `data/carprice.csv`, les données nettoyées sont traitées avec scikit-learn afin de créer un modèle de prédiction de prix.
Ce modèle est importé à l'aide de pickle `pickle/model.pkl` dans une application streamlit `streamlit/app.py`.

### Remarque :
- **Après nettoyage les données sont stockées dans `data/data_utilisable.csv`.**


## Structure du projet

Le projet est organisé en plusieurs dossiers :

- `data` : contient les fichiers csv.
- `notebook` : contient les notebooks Jupyter.
- `pickle` : contient le fichier pkl.
- `streamlit` : contient les fichiers Python et les composants Streamlit pour l'interface utilisateur de l'application.

## Installation

1. Clonez ce dépôt : `git clone git@github.com:Olaffson/prix_voiture.git`
2. Créez un environnement virtuel pour ce projet pour installer les dépendances .
3. Installez les dépendances : `pip install -r requirements.txt`

## Application 

Pour lancer l'application, il faut se placer dans le dossier `streamlit/app.py` et lancer la commande suivante :
 - `streamlit run app.py` pour extraire les données des films
 
## License
Ce projet est sous licence MIT - voir [LICENSE](LICENSE) pour plus de détails.

## Auteurs
* **Olivier Kotwica**[@Olaffson](https://github.com/Olaffson)

