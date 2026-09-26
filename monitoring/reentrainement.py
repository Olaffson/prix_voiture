"""
Réentraînement du modèle et remplacement du modèle en service s'il est meilleur.

1. Les données d'entraînement sont les lots les plus récents (fenêtre
   glissante), complétés si besoin par des véhicules des données d'origine :
   avec l'inflation, les anciens prix ne sont plus représentatifs.
2. Un nouveau modèle (challenger) est évalué par validation croisée sur ces
   données. Sur le dernier lot, ses prédictions sont comparées à celles du
   modèle en service (champion), qui n'a jamais vu ce lot.
3. Si le challenger fait mieux sur le dernier lot, il est entraîné sur toutes
   les données de la fenêtre et remplace le champion. L'ancien modèle, sa
   référence et ses données sont archivés dans modeles/archives/, puis la
   référence est recalculée pour le nouveau modèle.

Utilisation :
    python -m monitoring.reentrainement [--lots data/nouvelles]
"""
import argparse
import datetime
import json
import pickle
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

from monitoring.drift import RAPPORTS_PAR_DEFAUT, SEUILS_PAR_DEFAUT, charger_seuils
from monitoring.reference import REFERENCE_PAR_DEFAUT, construire_reference, sauvegarder_reference
from pipeline.entrainement import CIBLE, MODELE_PAR_DEFAUT, RANDOM_STATE, creer_modele, evaluer, mesurer, sauvegarder
from pipeline.nettoyage import nettoyer
from simulation.generer_lot import DOSSIER_PAR_DEFAUT as LOTS_PAR_DEFAUT

RACINE = Path(__file__).parent.parent
ORIGINE = RACINE / 'data' / 'carprice.csv'
DONNEES_MODELE_PAR_DEFAUT = RACINE / 'data' / 'donnees_modele.csv'
ARCHIVES_PAR_DEFAUT = RACINE / 'modeles' / 'archives'


def lister_lots(dossier: Path) -> list[Path]:
    """Lots du dossier, du plus ancien au plus récent."""
    return sorted(dossier.glob('lot_*.csv'))


def construire_fenetre(lots: list[Path], origine: pd.DataFrame, nb_min: int) -> tuple[pd.DataFrame, list[str]]:
    """
    Données d'entraînement : les lots les plus récents, complétés par des véhicules d'origine.

    Args:
        lots (list): fichiers des lots, du plus ancien au plus récent.
        origine (pd.DataFrame): données d'origine brutes (data/carprice.csv).
        nb_min (int): nombre minimal de véhicules.

    Returns:
        tuple: données nettoyées (le dernier lot en dernier) et origine de ces données.
    """
    retenus = []
    nb = 0
    for lot in reversed(lots):
        if nb >= nb_min:
            break
        retenus.insert(0, lot)
        nb += len(pd.read_csv(lot))

    morceaux = [pd.read_csv(lot) for lot in retenus]
    sources = [str(lot.relative_to(RACINE)) if lot.is_relative_to(RACINE) else str(lot) for lot in retenus]
    if nb < nb_min:
        complement = origine.sample(n=min(nb_min - nb, len(origine)), random_state=RANDOM_STATE)
        morceaux.insert(0, complement)
        sources.insert(0, f'data/carprice.csv ({len(complement)} véhicules)')

    fenetre = nettoyer(pd.concat(morceaux, ignore_index=True))
    return fenetre, sources


def evaluer_challenger(fenetre: pd.DataFrame, nb_dernier_lot: int, nb_plis: int) -> tuple[dict, dict]:
    """
    Performance d'un nouveau modèle par validation croisée sur la fenêtre.

    Chaque véhicule est prédit par un modèle entraîné sans lui.

    Returns:
        tuple: performance sur toute la fenêtre, et sur les véhicules du dernier lot.
    """
    X = fenetre.drop(CIBLE, axis=1)
    y = fenetre[CIBLE]
    plis = KFold(n_splits=nb_plis, shuffle=True, random_state=RANDOM_STATE)
    y_pred = cross_val_predict(creer_modele(), X, y, cv=plis)

    return mesurer(y, y_pred), mesurer(y[-nb_dernier_lot:], y_pred[-nb_dernier_lot:])


def archiver(modele: Path, reference: Path, donnees_modele: Path, dossier: Path) -> Path:
    """Copie le modèle en service, sa référence et ses données dans un dossier d'archive."""
    dossier.mkdir(parents=True, exist_ok=True)
    for fichier in (modele, reference, donnees_modele):
        shutil.copy2(fichier, dossier / fichier.name)
    return dossier


def reentrainer(lots: list[Path], seuils: dict, modele: Path = MODELE_PAR_DEFAUT,
                reference: Path = REFERENCE_PAR_DEFAUT, donnees_modele: Path = DONNEES_MODELE_PAR_DEFAUT,
                archives: Path = ARCHIVES_PAR_DEFAUT, date: str | None = None) -> dict:
    """
    Réentraîne le modèle et remplace le modèle en service si le nouveau est meilleur sur le dernier lot.

    Returns:
        dict: données utilisées, performances comparées et décision (remplacé ou non).
    """
    date = date or datetime.date.today().isoformat()
    s = seuils['reentrainement']

    fenetre, sources = construire_fenetre(lots, pd.read_csv(ORIGINE), s['nb_vehicules_min'])
    dernier_lot = nettoyer(pd.read_csv(lots[-1]))
    performance_fenetre, performance_challenger = evaluer_challenger(fenetre, len(dernier_lot), s['nb_plis'])

    with open(modele, 'rb') as file:
        champion = pickle.load(file)
    performance_champion = evaluer(champion, dernier_lot.drop(CIBLE, axis=1), dernier_lot[CIBLE])

    remplace = performance_challenger['mae'] < performance_champion['mae']
    resultat = {
        'date': date,
        'dernier_lot': lots[-1].name,
        'donnees': sources,
        'nb_vehicules': len(fenetre),
        'champion_dernier_lot': performance_champion,
        'challenger_dernier_lot': performance_challenger,
        'challenger_fenetre': performance_fenetre,
        'remplace': bool(remplace),
    }
    if not remplace:
        return resultat

    # archivage du modèle en service, puis remplacement par le nouveau modèle entraîné sur toute la fenêtre
    resultat['archive'] = str(archiver(modele, reference, donnees_modele, archives / lots[-1].stem))
    nouveau = creer_modele().fit(fenetre.drop(CIBLE, axis=1), fenetre[CIBLE])
    sauvegarder(nouveau, modele)
    fenetre.to_csv(donnees_modele, index=False)
    sauvegarder_reference(construire_reference(fenetre, nouveau, performance_fenetre, sources), reference)
    return resultat


def rapport_markdown(resultat: dict) -> str:
    """Met en forme le résultat d'un réentraînement."""
    champion, challenger = resultat['champion_dernier_lot'], resultat['challenger_dernier_lot']
    decision = 'nouveau modèle mis en service' if resultat['remplace'] else 'modèle en service conservé'
    lignes = [
        f"# Réentraînement du {resultat['date']}", '',
        f"**Décision : {decision}**", '',
        f"Données d'entraînement : {resultat['nb_vehicules']} véhicules.", '',
        *[f'- {source}' for source in resultat['donnees']], '',
        f"## Comparaison sur le dernier lot ({resultat['dernier_lot']})", '',
        '| Mesure | Modèle en service | Nouveau modèle |', '|---|---|---|',
        f"| MAE | {champion['mae']:.0f} $ | {challenger['mae']:.0f} $ |",
        f"| RMSE | {champion['rmse']:.0f} $ | {challenger['rmse']:.0f} $ |",
        f"| R² | {champion['r2']:.3f} | {challenger['r2']:.3f} |", '',
        'Le nouveau modèle est évalué par validation croisée : chaque véhicule est prédit par un modèle '
        "entraîné sans lui. Le modèle en service n'a jamais vu le dernier lot.",
    ]
    if resultat['remplace']:
        fenetre = resultat['challenger_fenetre']
        lignes += ['', f"Nouvelle performance de référence (validation croisée sur toute la fenêtre) : "
                       f"MAE {fenetre['mae']:.0f} $, R² {fenetre['r2']:.3f}.", '',
                   f"Ancien modèle archivé dans `{resultat['archive']}`."]
    return '\n'.join(lignes) + '\n'


def main():
    parser = argparse.ArgumentParser(description='Réentraîne le modèle sur les lots les plus récents.')
    parser.add_argument('--lots', type=Path, default=LOTS_PAR_DEFAUT, help='dossier des lots')
    parser.add_argument('--seuils', type=Path, default=SEUILS_PAR_DEFAUT)
    parser.add_argument('--rapports', type=Path, default=RAPPORTS_PAR_DEFAUT, help='dossier des rapports')
    args = parser.parse_args()

    lots = lister_lots(args.lots)
    if not lots:
        parser.error(f'aucun lot dans {args.lots}')
    resultat = reentrainer(lots, charger_seuils(args.seuils))

    args.rapports.mkdir(parents=True, exist_ok=True)
    nom = f'reentrainement_{lots[-1].stem}'
    (args.rapports / f'{nom}.md').write_text(rapport_markdown(resultat), encoding='utf-8')
    with open(args.rapports / f'{nom}.json', 'w', encoding='utf-8') as file:
        json.dump(resultat, file, ensure_ascii=False, indent=2)
        file.write('\n')

    champion, challenger = resultat['champion_dernier_lot'], resultat['challenger_dernier_lot']
    print(f"MAE sur le dernier lot : modèle en service {champion['mae']:.0f} $, nouveau modèle {challenger['mae']:.0f} $")
    print('Nouveau modèle mis en service' if resultat['remplace'] else 'Modèle en service conservé')
    print(f'Rapport : {args.rapports / nom}.md')


if __name__ == '__main__':
    main()
