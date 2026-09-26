import pickle
from pathlib import Path

import pandas as pd
import pytest

from pipeline.entrainement import entrainer, sauvegarder

DATA = Path(__file__).parent.parent / 'data'


@pytest.fixture(scope='module')
def resultat():
    return entrainer(pd.read_csv(DATA / 'data_utilisable.csv'))


def test_scores_du_modele(resultat):
    _, scores = resultat
    assert scores['test']['r2'] > 0.9


def test_entrainement_reproductible(resultat):
    _, scores = resultat
    _, scores_bis = entrainer(pd.read_csv(DATA / 'data_utilisable.csv'))
    assert scores == scores_bis


def test_modele_sauvegarde_et_recharge(resultat, tmp_path):
    model, _ = resultat
    X = pd.read_csv(DATA / 'data_utilisable.csv').drop('prix', axis=1).head(3)

    sauvegarder(model, tmp_path / 'model.pkl')
    with open(tmp_path / 'model.pkl', 'rb') as file:
        recharge = pickle.load(file)

    assert (recharge.predict(X) == model.predict(X)).all()
