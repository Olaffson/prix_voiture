from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).parent.parent / 'streamlit' / 'app.py'


def test_estimation():
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    app.button[0].click().run()

    assert not app.exception
    assert app.markdown[0].value.startswith('Le prix estimé pour ce véhicule est de :')


def test_modeles_de_la_marque_selectionnee():
    app = AppTest.from_file(str(APP), default_timeout=60).run()

    app.selectbox[0].set_value('bmw').run()

    assert set(app.selectbox[1].options) == {'320i', 'x1', 'x3', 'z4', 'x4', 'x5'}
