from pathlib import Path

import pandas as pd

from pipeline.nettoyage import nettoyer

DATA = Path(__file__).parent.parent / 'data'


def test_nettoyage_reproduit_data_utilisable(tmp_path):
    """Le nettoyage de carprice.csv doit redonner exactement data_utilisable.csv."""
    fichier = tmp_path / 'nettoye.csv'
    nettoyer(pd.read_csv(DATA / 'carprice.csv')).to_csv(fichier, index=False)

    assert fichier.read_bytes() == (DATA / 'data_utilisable.csv').read_bytes()


def test_nettoyage_ne_modifie_pas_les_donnees_brutes():
    brut = pd.read_csv(DATA / 'carprice.csv')
    copie = brut.copy()

    nettoyer(brut)

    pd.testing.assert_frame_equal(brut, copie)
