"""
Détection du drift d'un nouveau lot de données et décision de réentraînement.

Le lot est comparé à la référence (monitoring/reference.json) sur trois plans :
- performance du modèle sur le lot, grâce aux prix réels ;
- drift de chaque colonne, mesuré par le PSI (Population Stability Index) et
  confirmé par un test du khi-deux, qui tient compte de la taille du lot ;
- valeurs texte jamais vues par le modèle (nouvelle marque...).

Utilisation :
    python -m monitoring.drift <lot_brut.csv> [--rapports monitoring/rapports]

Le lot est au format de data/carprice.csv. Un rapport Markdown et un résultat
JSON sont enregistrés dans le dossier des rapports.
"""
import argparse
import json
import pickle
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.pipeline import Pipeline

from monitoring.reference import (REFERENCE_PAR_DEFAUT, charger_reference, proportions_numeriques,
                                  proportions_texte)
from pipeline.entrainement import CIBLE, MODELE_PAR_DEFAUT, evaluer
from pipeline.nettoyage import nettoyer

SEUILS_PAR_DEFAUT = Path(__file__).parent / 'seuils.toml'
RAPPORTS_PAR_DEFAUT = Path(__file__).parent / 'rapports'

REENTRAINER = 'réentraîner'
SURVEILLER = 'surveiller'
AUCUNE = 'aucune action'
LOT_TROP_PETIT = 'lot trop petit'

# Évite une division par zéro ou un logarithme de zéro quand un intervalle est vide
EPSILON = 1e-4


def charger_seuils(chemin: Path = SEUILS_PAR_DEFAUT) -> dict:
    with open(chemin, 'rb') as file:
        return tomllib.load(file)


def psi(attendues, observees) -> float:
    """
    Population Stability Index entre deux répartitions (listes de proportions).

    PSI = somme des (observée - attendue) × ln(observée / attendue) sur chaque intervalle.
    """
    attendues = np.clip(np.asarray(attendues, dtype=float), EPSILON, None)
    observees = np.clip(np.asarray(observees, dtype=float), EPSILON, None)
    return float(np.sum((observees - attendues) * np.log(observees / attendues)))


def p_valeur(attendues, nb_attendus: int, observees, nb_observes: int) -> float:
    """
    Probabilité d'observer un écart au moins aussi grand entre les deux répartitions par
    simple hasard (test du khi-deux). Plus le lot est petit, plus un écart peut être dû au hasard.
    """
    comptes = np.array([np.asarray(attendues) * nb_attendus, np.asarray(observees) * nb_observes])
    comptes = comptes[:, comptes.sum(axis=0) > 0]
    if comptes.shape[1] < 2:
        return 1.0
    return float(chi2_contingency(comptes).pvalue)


def mesures_par_colonne(lot: pd.DataFrame, reference: dict) -> dict:
    """PSI et p-valeur de chaque colonne du lot par rapport à la référence."""
    repartitions = {}
    for colonne, repartition in reference['numeriques'].items():
        if colonne in lot:
            observees = proportions_numeriques(lot[colonne], repartition['bornes'])
            repartitions[colonne] = (repartition['proportions'], observees, lot[colonne].notna().sum())
    for colonne, repartition in reference['texte'].items():
        if colonne in lot:
            observees = proportions_texte(lot[colonne].dropna(), repartition['categories'])
            repartitions[colonne] = (list(repartition['proportions'].values()), list(observees.values()),
                                     lot[colonne].notna().sum())

    return {
        colonne: {'psi': psi(attendues, observees),
                  'p_valeur': p_valeur(attendues, reference['nb_vehicules'], observees, nb)}
        for colonne, (attendues, observees, nb) in repartitions.items()
    }


def niveau_drift(mesure: dict, seuils: dict) -> str | None:
    """Niveau de drift d'une colonne : important, modéré, ou None s'il n'est pas notable ou pas significatif."""
    if mesure['p_valeur'] >= seuils['significativite']:
        return None
    if mesure['psi'] > seuils['psi_important']:
        return 'important'
    if mesure['psi'] > seuils['psi_modere']:
        return 'modéré'
    return None


def valeurs_inconnues(lot: pd.DataFrame, reference: dict) -> dict:
    """Valeurs texte du lot que le modèle n'a pas vues pendant son entraînement, par colonne."""
    resultat = {}
    for colonne, connues in reference['valeurs_connues'].items():
        inconnues = set(lot[colonne].dropna()) - set(connues)
        if inconnues:
            resultat[colonne] = sorted(inconnues)
    return resultat


def analyser_lot(lot: pd.DataFrame, reference: dict, model: Pipeline, seuils: dict) -> dict:
    """
    Compare un lot nettoyé à la référence et décide s'il faut réentraîner le modèle.

    Args:
        lot (pd.DataFrame): lot nettoyé (voir pipeline.nettoyage), avec la colonne prix.
        reference (dict): référence (voir monitoring.reference).
        model (Pipeline): modèle en service.
        seuils (dict): seuils de décision (voir monitoring/seuils.toml).

    Returns:
        dict: mesures, raisons et décision (réentraîner, surveiller, aucune action ou lot trop petit).
    """
    resultat = {'nb_vehicules': len(lot), 'raisons': [], 'alertes': []}
    if len(lot) < seuils['taille_minimale_lot']:
        resultat['decision'] = LOT_TROP_PETIT
        resultat['raisons'].append(f"{len(lot)} véhicules, minimum {seuils['taille_minimale_lot']}")
        return resultat

    # 1. performance du modèle sur le lot
    performance = evaluer(model, lot.drop(CIBLE, axis=1), lot[CIBLE])
    mae_max = seuils['performance']['facteur_mae_max'] * reference['performance']['mae']
    resultat['performance'] = performance
    resultat['mae_max'] = mae_max
    if performance['mae'] > mae_max:
        resultat['raisons'].append(f"erreur moyenne de {performance['mae']:.0f} $, au-delà de {mae_max:.0f} $")
    if performance['r2'] < seuils['performance']['r2_min']:
        resultat['raisons'].append(f"R² de {performance['r2']:.2f}, sous {seuils['performance']['r2_min']:.2f}")

    # 2. drift des colonnes
    s = seuils['drift']
    mesures = mesures_par_colonne(lot, reference)
    niveaux = {colonne: niveau_drift(mesure, s) for colonne, mesure in mesures.items()}
    en_drift = [c for c, niveau in niveaux.items() if niveau == 'important' and c in reference['importances']]
    part_importance = sum(reference['importances'][c] for c in en_drift)
    resultat['colonnes'] = {colonne: {**mesure, 'drift': niveaux[colonne]} for colonne, mesure in mesures.items()}
    resultat['colonnes_en_drift'] = en_drift
    resultat['part_importance_en_drift'] = part_importance
    if part_importance >= s['part_importance_max']:
        resultat['raisons'].append(f"colonnes en drift important ({', '.join(en_drift)}) représentant "
                                   f"{part_importance:.0%} de l'importance du modèle")
    for colonne, niveau in niveaux.items():
        if niveau and colonne not in en_drift:
            resultat['alertes'].append(f"drift {niveau} de {colonne} (PSI {mesures[colonne]['psi']:.2f})")

    # 3. valeurs jamais vues par le modèle
    s = seuils['valeurs_inconnues']
    inconnues = valeurs_inconnues(lot, reference)
    colonnes_suivies = [c for c in inconnues if c not in s['colonnes_ignorees']]
    concernes = pd.Series(False, index=lot.index)
    for colonne in colonnes_suivies:
        concernes |= lot[colonne].isin(inconnues[colonne])
    resultat['valeurs_inconnues'] = inconnues
    resultat['part_vehicules_inconnus'] = float(concernes.mean())
    if concernes.mean() > s['part_max']:
        detail = ', '.join(f"{c} : {', '.join(inconnues[c])}" for c in colonnes_suivies)
        resultat['raisons'].append(f"{concernes.mean():.0%} des véhicules ont une valeur inconnue du modèle ({detail})")
    elif colonnes_suivies:
        resultat['alertes'].append(f'valeurs inconnues du modèle sur {concernes.mean():.0%} des véhicules')

    if resultat['raisons']:
        resultat['decision'] = REENTRAINER
    elif resultat['alertes']:
        resultat['decision'] = SURVEILLER
    else:
        resultat['decision'] = AUCUNE
    return resultat


def rapport_markdown(nom_lot: str, resultat: dict, reference: dict, seuils: dict) -> str:
    """Met en forme le résultat de l'analyse d'un lot."""
    lignes = [f'# Rapport de drift : {nom_lot}', '', f"**Décision : {resultat['decision']}**", '']
    if resultat['raisons']:
        lignes += ['Raisons :', ''] + [f'- {raison}' for raison in resultat['raisons']] + ['']
    if resultat['alertes']:
        lignes += ['Alertes :', ''] + [f'- {alerte}' for alerte in resultat['alertes']] + ['']
    if resultat['decision'] == LOT_TROP_PETIT:
        return '\n'.join(lignes)

    ref, lot = reference['performance'], resultat['performance']
    lignes += [
        '## Performance du modèle', '',
        '| Mesure | Référence | Lot | Seuil |', '|---|---|---|---|',
        f"| MAE | {ref['mae']:.0f} $ | {lot['mae']:.0f} $ | {resultat['mae_max']:.0f} $ |",
        f"| RMSE | {ref['rmse']:.0f} $ | {lot['rmse']:.0f} $ | |",
        f"| R² | {ref['r2']:.3f} | {lot['r2']:.3f} | {seuils['performance']['r2_min']:.2f} |",
        '', '## Drift des colonnes', '',
        f"Colonnes en drift important : {resultat['part_importance_en_drift']:.0%} de l'importance du modèle "
        f"(seuil {seuils['drift']['part_importance_max']:.0%}). Un drift n'est retenu que si la p-valeur "
        f"est inférieure à {seuils['drift']['significativite']}.", '',
        '| Colonne | Importance | PSI | p-valeur | Drift |', '|---|---|---|---|---|',
    ]
    for colonne, mesure in sorted(resultat['colonnes'].items(), key=lambda x: -x[1]['psi']):
        importance = reference['importances'].get(colonne)
        importance = f'{importance:.1%}' if importance is not None else '(prix)'
        lignes.append(f"| {colonne} | {importance} | {mesure['psi']:.2f} | {mesure['p_valeur']:.3f} | "
                      f"{mesure['drift'] or '-'} |")

    lignes += ['', '## Valeurs inconnues du modèle', '']
    if resultat['valeurs_inconnues']:
        lignes += [f"- {colonne} : {', '.join(map(str, valeurs))}" for colonne, valeurs in resultat['valeurs_inconnues'].items()]
        lignes += ['', f"Véhicules concernés (hors {', '.join(seuils['valeurs_inconnues']['colonnes_ignorees'])}) : "
                       f"{resultat['part_vehicules_inconnus']:.0%}."]
    else:
        lignes.append('Aucune.')
    return '\n'.join(lignes) + '\n'


def analyser_fichier(lot: Path, reference: Path = REFERENCE_PAR_DEFAUT, modele: Path = MODELE_PAR_DEFAUT,
                     seuils: Path = SEUILS_PAR_DEFAUT, rapports: Path = RAPPORTS_PAR_DEFAUT) -> tuple[dict, Path]:
    """
    Analyse un lot brut et enregistre son rapport (Markdown et JSON) dans le dossier des rapports.

    Returns:
        tuple: résultat de l'analyse et chemin du rapport Markdown.
    """
    reference, seuils = charger_reference(reference), charger_seuils(seuils)
    with open(modele, 'rb') as file:
        model = pickle.load(file)
    resultat = analyser_lot(nettoyer(pd.read_csv(lot)), reference, model, seuils)

    rapports.mkdir(parents=True, exist_ok=True)
    rapport = rapports / f'{lot.stem}.md'
    rapport.write_text(rapport_markdown(lot.stem, resultat, reference, seuils), encoding='utf-8')
    with open(rapport.with_suffix('.json'), 'w', encoding='utf-8') as file:
        json.dump({'lot': lot.name, **resultat}, file, ensure_ascii=False, indent=2)
        file.write('\n')
    return resultat, rapport


def main():
    parser = argparse.ArgumentParser(description="Analyse le drift d'un lot de données.")
    parser.add_argument('lot', type=Path, help='lot au format de data/carprice.csv')
    parser.add_argument('--reference', type=Path, default=REFERENCE_PAR_DEFAUT)
    parser.add_argument('--modele', type=Path, default=MODELE_PAR_DEFAUT)
    parser.add_argument('--seuils', type=Path, default=SEUILS_PAR_DEFAUT)
    parser.add_argument('--rapports', type=Path, default=RAPPORTS_PAR_DEFAUT, help='dossier des rapports')
    args = parser.parse_args()

    resultat, rapport = analyser_fichier(args.lot, args.reference, args.modele, args.seuils, args.rapports)

    print(f"{args.lot.stem} : {resultat['decision']}")
    for raison in resultat['raisons']:
        print(f'  - {raison}')
    print(f'Rapport : {rapport}')


if __name__ == '__main__':
    main()
