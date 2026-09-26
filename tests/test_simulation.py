import copy
from pathlib import Path

import pandas as pd
import pytest

from pipeline.nettoyage import nettoyer
from simulation.generer_lot import charger_scenario, generer_lot, prochain_numero

DATA = Path(__file__).parent.parent / 'data'


@pytest.fixture(scope='module')
def origine():
    return pd.read_csv(DATA / 'carprice.csv')


@pytest.fixture
def scenario():
    return charger_scenario()


@pytest.fixture
def grand_scenario(scenario):
    """Scénario avec de grands lots, pour que les moyennes soient stables."""
    scenario = copy.deepcopy(scenario)
    scenario['taille_lot'] = 3000
    return scenario


def test_lot_reproductible(scenario, origine):
    pd.testing.assert_frame_equal(generer_lot(3, scenario, origine), generer_lot(3, scenario, origine))


def test_lots_differents(scenario, origine):
    assert not generer_lot(1, scenario, origine).equals(generer_lot(2, scenario, origine))


def test_format_identique_aux_donnees_origine(scenario, origine):
    lot = generer_lot(5, scenario, origine)

    assert len(lot) == scenario['taille_lot']
    assert list(lot.columns) == list(origine.columns)
    assert (lot.dtypes == origine.dtypes).all()


def test_lot_nettoyable(scenario, origine):
    propre = nettoyer(generer_lot(8, scenario, origine))

    assert propre.drop('modele', axis=1).notna().all().all()


def test_sans_drift_proche_des_donnees_origine(scenario, origine):
    scenario['taille_lot'] = 3000
    scenario['inflation']['taux_par_lot'] = 0
    scenario['puissance']['taux_par_lot'] = 0
    scenario['gamme']['hausse_par_lot'] = 0
    scenario['relation_puissance_prix']['intensite_par_lot'] = 0
    scenario['nouvelles_marques'] = []

    lot = generer_lot(10, scenario, origine)

    for colonne in ['price', 'horsepower', 'curbweight']:
        assert lot[colonne].mean() == pytest.approx(origine[colonne].mean(), rel=0.05)


def test_drift_augmente_avec_le_numero(grand_scenario, origine):
    lot_1 = generer_lot(1, grand_scenario, origine)
    lot_8 = generer_lot(8, grand_scenario, origine)

    assert lot_8['price'].mean() > lot_1['price'].mean() * 1.1
    assert lot_8['horsepower'].mean() > lot_1['horsepower'].mean() * 1.05
    assert (lot_8['fueltype'] == 'diesel').mean() > (lot_1['fueltype'] == 'diesel').mean()


def test_nouvelles_marques(scenario, origine):
    tesla = scenario['nouvelles_marques'][0]
    debut = tesla['a_partir_du_lot']

    avant = generer_lot(debut - 1, scenario, origine)
    apres = generer_lot(debut, scenario, origine)

    assert not avant['CarName'].str.startswith(tesla['nom']).any()
    assert apres['CarName'].str.startswith(tesla['nom']).sum() == round(tesla['part'] * scenario['taille_lot'])
    assert (nettoyer(apres)['marque'] == tesla['nom']).any()


def test_prochain_numero(tmp_path):
    assert prochain_numero(tmp_path) == 1

    (tmp_path / 'lot_001_2026-10-05.csv').touch()
    (tmp_path / 'lot_002_2026-10-12.csv').touch()

    assert prochain_numero(tmp_path) == 3
