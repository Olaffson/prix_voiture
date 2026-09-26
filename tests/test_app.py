from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

RACINE = Path(__file__).parent.parent
APP = RACINE / 'streamlit' / 'app.py'


def test_estimation():
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    app.button[0].click().run()

    assert not app.exception
    assert app.markdown[0].value.startswith('Le prix estimé pour ce véhicule est de :')


def test_modeles_de_la_marque_selectionnee():
    donnees = pd.read_csv(RACINE / 'data' / 'donnees_modele.csv')
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    for marque in app.selectbox[0].options[:3]:
        app.selectbox[0].set_value(marque).run()

        attendus = set(donnees.loc[donnees['marque'] == marque, 'modele'].dropna())
        assert set(app.selectbox[1].options) == attendus
