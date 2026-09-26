"""
Cycle hebdomadaire : nouveau lot, analyse du drift et réentraînement si besoin.

Utilisation :
    python -m monitoring.cycle [--date AAAA-MM-JJ]

Dans une GitHub Action, le résultat est transmis aux étapes suivantes
(variable GITHUB_OUTPUT) et les rapports sont affichés dans le résumé de
l'exécution (variable GITHUB_STEP_SUMMARY) :
- lot : nom du lot ;
- decision : décision de l'analyse du drift ;
- remplace : true si un nouveau modèle a été mis en service ;
- fichiers_main : fichiers à enregistrer directement sur la branche principale
  (lot et rapports) ;
- rapport_reentrainement : rapport du réentraînement, qui sert de description
  à la pull request quand un nouveau modèle a été mis en service.
"""
import argparse
import datetime
import os
from pathlib import Path

import pandas as pd

from monitoring.drift import REENTRAINER, RAPPORTS_PAR_DEFAUT, SEUILS_PAR_DEFAUT, analyser_fichier, charger_seuils
from monitoring.reentrainement import (ARCHIVES_PAR_DEFAUT, DONNEES_MODELE_PAR_DEFAUT, _chemin_lisible, lister_lots,
                                       reentrainer_et_rapporter)
from monitoring.reference import REFERENCE_PAR_DEFAUT
from pipeline.entrainement import MODELE_PAR_DEFAUT
from simulation.generer_lot import (DONNEES_ORIGINE, DOSSIER_PAR_DEFAUT, SCENARIO_PAR_DEFAUT, charger_scenario,
                                    generer_lot, prochain_numero)


def executer_cycle(date: str, lots: Path = DOSSIER_PAR_DEFAUT, rapports: Path = RAPPORTS_PAR_DEFAUT,
                   scenario: Path = SCENARIO_PAR_DEFAUT, seuils: Path = SEUILS_PAR_DEFAUT,
                   modele: Path = MODELE_PAR_DEFAUT, reference: Path = REFERENCE_PAR_DEFAUT,
                   donnees_modele: Path = DONNEES_MODELE_PAR_DEFAUT, archives: Path = ARCHIVES_PAR_DEFAUT) -> dict:
    """
    Génère le lot de la semaine, l'analyse, et réentraîne le modèle si l'analyse le demande.

    Returns:
        dict: lot, décision, réentraînement, fichiers à enregistrer et rapports.
    """
    # 1. nouveau lot
    lots.mkdir(parents=True, exist_ok=True)
    numero = prochain_numero(lots)
    lot = lots / f'lot_{numero:03d}_{date}.csv'
    generer_lot(numero, charger_scenario(scenario), pd.read_csv(DONNEES_ORIGINE)).to_csv(lot, index=False)

    # 2. analyse du drift
    analyse, rapport_drift = analyser_fichier(lot, reference, modele, seuils, rapports)
    resume = {
        'lot': lot.stem,
        'decision': analyse['decision'],
        'remplace': False,
        'fichiers_main': [lot, rapport_drift, rapport_drift.with_suffix('.json')],
        'rapports': [rapport_drift],
        'rapport_reentrainement': None,
    }

    # 3. réentraînement si besoin
    if analyse['decision'] == REENTRAINER:
        resultat, rapport = reentrainer_et_rapporter(lister_lots(lots), charger_seuils(seuils), rapports,
                                                     modele=modele, reference=reference,
                                                     donnees_modele=donnees_modele, archives=archives)
        resume['remplace'] = resultat['remplace']
        resume['rapports'].append(rapport)
        resume['rapport_reentrainement'] = rapport
        if not resultat['remplace']:
            # modèle conservé : le rapport est enregistré avec le lot. Sinon, il accompagne le nouveau modèle.
            resume['fichiers_main'] += [rapport, rapport.with_suffix('.json')]
    return resume


def ecrire_sorties_github(resume: dict) -> None:
    """Transmet le résultat aux étapes suivantes d'une GitHub Action et affiche les rapports dans son résumé."""
    if 'GITHUB_OUTPUT' in os.environ:
        rapport = resume['rapport_reentrainement']
        with open(os.environ['GITHUB_OUTPUT'], 'a', encoding='utf-8') as sortie:
            sortie.write(f"lot={resume['lot']}\n")
            sortie.write(f"decision={resume['decision']}\n")
            sortie.write(f"remplace={'true' if resume['remplace'] else 'false'}\n")
            sortie.write(f"fichiers_main={' '.join(_chemin_lisible(f) for f in resume['fichiers_main'])}\n")
            sortie.write(f"rapport_reentrainement={_chemin_lisible(rapport) if rapport else ''}\n")
    if 'GITHUB_STEP_SUMMARY' in os.environ:
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a', encoding='utf-8') as resume_github:
            for rapport in resume['rapports']:
                resume_github.write(rapport.read_text(encoding='utf-8') + '\n')


def main():
    parser = argparse.ArgumentParser(description='Cycle hebdomadaire : nouveau lot, drift et réentraînement.')
    parser.add_argument('--date', default=datetime.date.today().isoformat(), help='date du lot (AAAA-MM-JJ)')
    args = parser.parse_args()

    resume = executer_cycle(args.date)
    ecrire_sorties_github(resume)

    print(f"{resume['lot']} : {resume['decision']}")
    if resume['rapport_reentrainement']:
        print('Nouveau modèle mis en service' if resume['remplace'] else 'Modèle en service conservé')


if __name__ == '__main__':
    main()
