from pathlib import Path

import pandas as pd
import pytest

from monitoring.cycle import ecrire_sorties_github, executer_cycle
from monitoring.reference import construire_reference, sauvegarder_reference
from pipeline.entrainement import entrainer, sauvegarder

RACINE = Path(__file__).parent.parent


@pytest.fixture(scope='module')
def modele_origine():
    donnees = pd.read_csv(RACINE / 'data' / 'data_utilisable.csv')
    model, _ = entrainer(donnees)
    return model, construire_reference(donnees, model), donnees


@pytest.fixture
def chemins(tmp_path, modele_origine):
    """Modèle d'origine, sa référence et ses données, et dossiers du cycle, dans un dossier temporaire."""
    chemins = {
        'lots': tmp_path / 'nouvelles',
        'rapports': tmp_path / 'rapports',
        'modele': tmp_path / 'model.pkl',
        'reference': tmp_path / 'reference.json',
        'donnees_modele': tmp_path / 'donnees_modele.csv',
        'archives': tmp_path / 'archives',
    }
    model, reference, donnees = modele_origine
    sauvegarder(model, chemins['modele'])
    sauvegarder_reference(reference, chemins['reference'])
    donnees.to_csv(chemins['donnees_modele'], index=False)
    return chemins


def test_premier_cycle(chemins):
    resume = executer_cycle('2026-10-05', **chemins)

    assert resume['lot'] == 'lot_001_2026-10-05'
    assert (chemins['lots'] / 'lot_001_2026-10-05.csv').exists()
    assert resume['fichiers_main'] == [
        chemins['lots'] / 'lot_001_2026-10-05.csv',
        chemins['rapports'] / 'lot_001_2026-10-05.md',
        chemins['rapports'] / 'lot_001_2026-10-05.json',
    ]
    assert all(fichier.exists() for fichier in resume['fichiers_main'])


def test_cycles_successifs(chemins):
    ancien_modele = chemins['modele'].read_bytes()

    resumes = [executer_cycle(f'2026-10-0{semaine}', **chemins) for semaine in range(1, 6)]

    assert [r['lot'][:7] for r in resumes] == ['lot_001', 'lot_002', 'lot_003', 'lot_004', 'lot_005']
    # le drift du scénario impose au moins un nouveau modèle en 5 semaines (voir simulation/scenario.toml)
    assert any(r['remplace'] for r in resumes)
    assert chemins['modele'].read_bytes() != ancien_modele
    for resume in resumes:
        rapport = resume['rapport_reentrainement']
        if resume['decision'] != 'réentraîner':
            assert rapport is None and not resume['remplace']
        elif resume['remplace']:
            # le rapport accompagne le nouveau modèle dans la pull request
            assert rapport.exists() and rapport not in resume['fichiers_main']
        else:
            # modèle conservé : le rapport est enregistré avec le lot
            assert rapport in resume['fichiers_main']


def test_sorties_github(chemins, tmp_path, monkeypatch):
    sortie, resume_github = tmp_path / 'sortie', tmp_path / 'resume.md'
    monkeypatch.setenv('GITHUB_OUTPUT', str(sortie))
    monkeypatch.setenv('GITHUB_STEP_SUMMARY', str(resume_github))

    resume = executer_cycle('2026-10-05', **chemins)
    ecrire_sorties_github(resume)

    valeurs = dict(ligne.split('=', 1) for ligne in sortie.read_text(encoding='utf-8').splitlines())
    assert valeurs['lot'] == 'lot_001_2026-10-05'
    assert valeurs['decision'] == resume['decision']
    assert valeurs['remplace'] == 'false'
    assert valeurs['fichiers_main'].split() == [str(f) for f in resume['fichiers_main']]
    assert valeurs['rapport_reentrainement'] == ''
    assert '# Rapport de drift : lot_001_2026-10-05' in resume_github.read_text(encoding='utf-8')
