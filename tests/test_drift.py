import json
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from monitoring.drift import (AUCUNE, LOT_TROP_PETIT, REENTRAINER, analyser_lot, charger_seuils, p_valeur, psi,
                              rapport_markdown)
from monitoring.reference import charger_reference, construire_reference
from pipeline.entrainement import MODELE_PAR_DEFAUT
from pipeline.nettoyage import nettoyer

RACINE = Path(__file__).parent.parent
DATA = RACINE / 'data'


@pytest.fixture(scope='module')
def model():
    with open(MODELE_PAR_DEFAUT, 'rb') as file:
        return pickle.load(file)


@pytest.fixture(scope='module')
def donnees():
    return pd.read_csv(DATA / 'data_utilisable.csv')


@pytest.fixture(scope='module')
def reference(donnees, model):
    return construire_reference(donnees, model)


@pytest.fixture
def seuils():
    return charger_seuils()


@pytest.fixture
def lot(donnees):
    """Lot sans drift : 100 véhicules tirés des données d'entraînement."""
    return donnees.sample(n=100, random_state=0).reset_index(drop=True)


def test_psi():
    assert psi([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]) == pytest.approx(0)
    assert psi([0.2, 0.3, 0.5], [0.5, 0.3, 0.2]) > 0.25


def test_p_valeur_tient_compte_de_la_taille_du_lot():
    attendues, observees = [0.5, 0.5], [0.6, 0.4]

    assert p_valeur(attendues, 200, observees, 20) > 0.1
    assert p_valeur(attendues, 2000, observees, 2000) < 0.01


def test_reference(reference):
    assert sum(reference['importances'].values()) == pytest.approx(1)
    assert 'prix' not in reference['importances']
    assert 'toyota' in reference['valeurs_connues']['marque']
    assert reference['performance']['r2'] > 0.9
    # modele n'a que des valeurs rares : il n'est pas suivi par le PSI
    assert 'modele' not in reference['texte']


def test_reference_enregistree_a_jour(reference):
    """monitoring/reference.json doit correspondre au modèle en service."""
    enregistree = charger_reference()
    # aller-retour en JSON pour comparer les mêmes types (listes, flottants)
    reference = json.loads(json.dumps(reference))
    for cle in ['performance', 'importances', 'valeurs_connues', 'numeriques', 'texte']:
        assert enregistree[cle] == reference[cle]


def test_lot_sans_drift(lot, reference, model, seuils):
    resultat = analyser_lot(lot, reference, model, seuils)

    assert resultat['decision'] != REENTRAINER
    assert resultat['colonnes_en_drift'] == []


def test_lot_trop_petit(lot, reference, model, seuils):
    resultat = analyser_lot(lot.head(10), reference, model, seuils)

    assert resultat['decision'] == LOT_TROP_PETIT


def test_prix_en_hausse(lot, reference, model, seuils):
    lot['prix'] *= 1.5

    resultat = analyser_lot(lot, reference, model, seuils)

    assert resultat['decision'] == REENTRAINER
    assert any('erreur moyenne' in raison for raison in resultat['raisons'])
    assert resultat['colonnes']['prix']['drift'] is not None


def test_nouvelle_marque(lot, reference, model, seuils):
    lot.loc[:19, 'marque'] = 'tesla'

    resultat = analyser_lot(lot, reference, model, seuils)

    assert resultat['decision'] == REENTRAINER
    assert resultat['valeurs_inconnues']['marque'] == ['tesla']
    assert resultat['part_vehicules_inconnus'] >= 0.2


def test_modele_inconnu_ignore(lot, reference, model, seuils):
    lot.loc[:19, 'modele'] = 'modele inconnu'

    resultat = analyser_lot(lot, reference, model, seuils)

    assert 'modele inconnu' in resultat['valeurs_inconnues']['modele']
    assert resultat['part_vehicules_inconnus'] < seuils['valeurs_inconnues']['part_max']


def test_drift_colonne_importante(lot, reference, model, seuils):
    lot['taille_moteur'] = lot['taille_moteur'].max() * 2

    resultat = analyser_lot(lot, reference, model, seuils)

    assert 'taille_moteur' in resultat['colonnes_en_drift']
    assert resultat['decision'] == REENTRAINER


def test_rapport(lot, reference, model, seuils):
    resultat = analyser_lot(lot, reference, model, seuils)

    rapport = rapport_markdown('lot_test', resultat, reference, seuils)

    assert f"**Décision : {resultat['decision']}**" in rapport
    assert '| taille_moteur |' in rapport


def test_ligne_de_commande(tmp_path):
    fichier = tmp_path / 'lot_001_2026-10-05.csv'
    pd.read_csv(DATA / 'carprice.csv').sample(n=80, random_state=1).to_csv(fichier, index=False)

    subprocess.run([sys.executable, '-m', 'monitoring.drift', str(fichier), '--rapports', str(tmp_path / 'rapports')],
                   cwd=RACINE, check=True, capture_output=True)

    resultat = json.loads((tmp_path / 'rapports' / 'lot_001_2026-10-05.json').read_text(encoding='utf-8'))
    assert resultat['decision'] in (AUCUNE, 'surveiller')
    assert (tmp_path / 'rapports' / 'lot_001_2026-10-05.md').exists()


def test_lot_brut_nettoye(reference, model, seuils):
    brut = pd.read_csv(DATA / 'carprice.csv').sample(n=80, random_state=1)

    resultat = analyser_lot(nettoyer(brut), reference, model, seuils)

    assert resultat['decision'] != REENTRAINER
