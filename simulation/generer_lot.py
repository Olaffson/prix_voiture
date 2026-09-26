"""
Génération de lots de données simulés, au format de data/carprice.csv.

Chaque lot est tiré des véhicules d'origine, avec un léger bruit, puis
déformé selon le scénario de drift de simulation/scenario.toml. Plus le
numéro du lot est élevé, plus le drift est important.

Utilisation :
    python -m simulation.generer_lot [--numero N] [--date AAAA-MM-JJ] [--dossier data/nouvelles]

Par défaut, le numéro est celui qui suit le dernier lot du dossier et la
date est celle du jour. Le lot est enregistré dans <dossier>/lot_NNN_AAAA-MM-JJ.csv.
"""
import argparse
import datetime
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

RACINE = Path(__file__).parent.parent
DONNEES_ORIGINE = RACINE / 'data' / 'carprice.csv'
SCENARIO_PAR_DEFAUT = Path(__file__).parent / 'scenario.toml'
DOSSIER_PAR_DEFAUT = RACINE / 'data' / 'nouvelles'

# Colonnes numériques bruitées, avec le nombre de décimales des données d'origine
COLONNES_BRUITEES = {
    'wheelbase': 1,
    'carlength': 1,
    'carwidth': 1,
    'carheight': 1,
    'curbweight': 0,
    'enginesize': 0,
    'boreratio': 2,
    'stroke': 2,
    'compressionratio': 2,
    'peakrpm': 0,
    'citympg': 0,
    'highwaympg': 0,
}


def charger_scenario(chemin: Path = SCENARIO_PAR_DEFAUT) -> dict:
    """Lit le scénario de drift."""
    with open(chemin, 'rb') as file:
        return tomllib.load(file)


def _arrondir(serie: pd.Series, decimales: int) -> pd.Series:
    serie = serie.round(decimales)
    return serie.astype(int) if decimales == 0 else serie


def generer_lot(numero: int, scenario: dict, origine: pd.DataFrame) -> pd.DataFrame:
    """
    Génère un lot simulé.

    Args:
        numero (int): numéro du lot (1 pour le premier) : l'intensité du drift en dépend.
        scenario (dict): scénario de drift (voir simulation/scenario.toml).
        origine (pd.DataFrame): données d'origine, au format de data/carprice.csv.

    Returns:
        pd.DataFrame: le lot, au même format que les données d'origine.
    """
    rng = np.random.default_rng(scenario['graine'] + numero)
    taille = scenario['taille_lot']

    # 1. tirage des véhicules, en favorisant de plus en plus certaines caractéristiques (changement de gamme)
    gamme = scenario['gamme']
    poids = np.ones(len(origine))
    for colonne, valeur in gamme['preferences'].items():
        poids *= np.where(origine[colonne] == valeur, 1 + gamme['hausse_par_lot'] * numero, 1)
    lot = origine.sample(n=taille, replace=True, weights=poids, random_state=rng).reset_index(drop=True)

    # 2. bruit sur les caractéristiques numériques
    for colonne, decimales in COLONNES_BRUITEES.items():
        lot[colonne] = _arrondir(lot[colonne] * rng.normal(1, scenario['bruit_caracteristiques'], taille), decimales)

    # 3. moteurs plus puissants
    lot['horsepower'] = _arrondir(lot['horsepower'] * (1 + scenario['puissance']['taux_par_lot']) ** numero, 0)

    # 4. prix : la puissance pèse de plus en plus (concept drift), puis inflation et bruit
    ref = origine['horsepower']
    ecart_puissance = (lot['horsepower'] - ref.mean()) / ref.std()
    effet_puissance = (1 + scenario['relation_puissance_prix']['intensite_par_lot'] * numero * ecart_puissance).clip(lower=0.5)
    inflation = (1 + scenario['inflation']['taux_par_lot']) ** numero
    lot['price'] = lot['price'] * effet_puissance * inflation * rng.normal(1, scenario['bruit_prix'], taille)

    # 5. nouvelles marques : une partie des véhicules est renommée, avec un prix ajusté
    deja_renommes = np.zeros(taille, dtype=bool)
    for marque in scenario.get('nouvelles_marques', []):
        if numero < marque['a_partir_du_lot']:
            continue
        candidats = np.flatnonzero(~deja_renommes)
        choisis = rng.choice(candidats, size=round(marque['part'] * taille), replace=False)
        deja_renommes[choisis] = True
        modeles = rng.choice(marque['modeles'], size=len(choisis))
        lot.loc[choisis, 'CarName'] = [f"{marque['nom']} {modele}" for modele in modeles]
        lot.loc[choisis, 'price'] *= marque['facteur_prix']

    lot['price'] = lot['price'].round(0)
    lot['car_ID'] = range(numero * 1000 + 1, numero * 1000 + taille + 1)
    return lot


def prochain_numero(dossier: Path) -> int:
    """Numéro du lot qui suit le dernier lot présent dans le dossier."""
    numeros = [int(f.name.split('_')[1]) for f in dossier.glob('lot_*.csv')]
    return max(numeros, default=0) + 1


def main():
    parser = argparse.ArgumentParser(description='Génère un lot de données simulé.')
    parser.add_argument('--numero', type=int, help='numéro du lot (par défaut : le suivant dans le dossier)')
    parser.add_argument('--date', default=datetime.date.today().isoformat(), help='date du lot (AAAA-MM-JJ)')
    parser.add_argument('--dossier', type=Path, default=DOSSIER_PAR_DEFAUT, help='dossier de destination')
    parser.add_argument('--scenario', type=Path, default=SCENARIO_PAR_DEFAUT, help='scénario de drift')
    args = parser.parse_args()

    args.dossier.mkdir(parents=True, exist_ok=True)
    numero = args.numero or prochain_numero(args.dossier)

    lot = generer_lot(numero, charger_scenario(args.scenario), pd.read_csv(DONNEES_ORIGINE))
    fichier = args.dossier / f'lot_{numero:03d}_{args.date}.csv'
    lot.to_csv(fichier, index=False)
    print(f'Lot {numero} enregistré dans {fichier} ({len(lot)} véhicules)')


if __name__ == '__main__':
    main()
