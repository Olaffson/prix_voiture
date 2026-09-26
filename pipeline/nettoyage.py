"""
Nettoyage des données brutes (format de data/carprice.csv).

Reprend les étapes de notebook/nettoyage.ipynb pour pouvoir les appliquer
à n'importe quel nouveau fichier de données au même format.

Utilisation :
    python -m pipeline.nettoyage data/carprice.csv data/data_utilisable.csv
"""
import sys

import pandas as pd

# Correction des fautes d'orthographe sur les marques
CORRECTIONS_MARQUES = {
    'alfa-romero': 'alfa-romeo',
    'maxda': 'mazda',
    'toyouta': 'toyota',
    'vokswagen': 'volkswagen',
    'vw': 'volkswagen',
    'Nissan': 'nissan',
    'porcshce': 'porsche',
}

# Correspondance entre les noms de colonnes anglais et français
COLONNES_FR = {
    'symboling': 'risque_assurance',
    'fueltype': 'carburant',
    'aspiration': 'turbo',
    'doornumber': 'nombre_portes',
    'carbody': 'type_vehicule',
    'drivewheel': 'roues_motrices',
    'enginelocation': 'emplacement_moteur',
    'enginesize': 'taille_moteur',
    'wheelbase': 'empattement',
    'carlength': 'longueur_voiture',
    'carwidth': 'largeur_voiture',
    'carheight': 'hauteur_voiture',
    'curbweight': 'poids_voiture',
    'enginetype': 'type_moteur',
    'cylindernumber': 'nombre_cylindres',
    'fuelsystem': 'systeme_carburant',
    'boreratio': 'taux_alesage',
    'stroke': 'course',
    'compressionratio': 'taux_compression',
    'horsepower': 'puissance',
    'peakrpm': 'tour_moteur',
    'citympg': 'consommation_ville',
    'highwaympg': 'consommation_autoroute',
    'price': 'prix',
}

POUCES_EN_CM = 2.54
LIVRES_EN_KG = 0.453592
# consommation (L/100 km) = MPG_EN_L100KM / consommation (miles par gallon)
MPG_EN_L100KM = 235.215


def nettoyer(df_brut: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie des données brutes au format de data/carprice.csv.

    Args:
        df_brut (pd.DataFrame): données brutes (colonnes en anglais, unités américaines).

    Returns:
        pd.DataFrame: données nettoyées (colonnes en français, unités métriques,
        colonnes marque et modele à la place de carname).
    """
    df = df_brut.copy()

    # nettoyage basique des noms de colonnes
    df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

    # retrait de l'identifiant, inutile pour la prédiction
    df = df.drop('car_id', axis=1)

    # conversion des dimensions de pouces en centimètres
    # (round de Python, et non Series.round, pour arrondir exactement comme le notebook d'origine)
    for col in ['wheelbase', 'carlength', 'carwidth', 'carheight']:
        df[col] = df[col].apply(lambda x: round(x * POUCES_EN_CM, 1))

    # conversion du poids de livres en kg
    df['curbweight'] = df['curbweight'] * LIVRES_EN_KG

    # conversion des consommations de miles/gallon en litres/100 km
    for col in ['citympg', 'highwaympg']:
        df[col] = MPG_EN_L100KM / df[col]

    # création des colonnes marque et modele
    df[['marque', 'modele']] = df['carname'].str.split(' ', n=1, expand=True)
    df = df.drop('carname', axis=1)
    df['marque'] = df['marque'].replace(CORRECTIONS_MARQUES)

    # renommage des colonnes en français
    return df.rename(columns=COLONNES_FR)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.exit('Utilisation : python -m pipeline.nettoyage <fichier_brut.csv> <fichier_nettoye.csv>')
    nettoyer(pd.read_csv(sys.argv[1])).to_csv(sys.argv[2], index=False)
