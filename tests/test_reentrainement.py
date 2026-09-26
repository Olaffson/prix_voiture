import pickle
import shutil
from pathlib import Path

import pandas as pd
import pytest

from monitoring.drift import charger_seuils
from monitoring.reentrainement import construire_fenetre, reentrainer, rapport_markdown
from monitoring.reference import charger_reference, construire_reference, sauvegarder_reference
from pipeline.entrainement import CIBLE, creer_modele, entrainer, sauvegarder
from simulation.generer_lot import charger_scenario, generer_lot

RACINE = Path(__file__).parent.parent
ORIGINE = pd.read_csv(RACINE / 'data' / 'carprice.csv')


def creer_lots(dossier: Path, numeros) -> list[Path]:
    dossier.mkdir(exist_ok=True)
    scenario = charger_scenario()
    lots = []
    for numero in numeros:
        fichier = dossier / f'lot_{numero:03d}_2026-10-{numero:02d}.csv'
        generer_lot(numero, scenario, ORIGINE).to_csv(fichier, index=False)
        lots.append(fichier)
    return lots


@pytest.fixture(scope='module')
def modele_origine():
    donnees = pd.read_csv(RACINE / 'data' / 'data_utilisable.csv')
    model, _ = entrainer(donnees)
    return model, construire_reference(donnees, model)


@pytest.fixture
def modele_en_service(tmp_path, modele_origine):
    """
    Modèle d'origine, sa référence et ses données, dans un dossier temporaire.

    Les tests ne dépendent pas du modèle réellement en service, qui change à chaque réentraînement.
    """
    chemins = {
        'modele': tmp_path / 'model.pkl',
        'reference': tmp_path / 'reference.json',
        'donnees_modele': tmp_path / 'donnees_modele.csv',
        'archives': tmp_path / 'archives',
    }
    model, reference = modele_origine
    sauvegarder(model, chemins['modele'])
    sauvegarder_reference(reference, chemins['reference'])
    shutil.copy(RACINE / 'data' / 'data_utilisable.csv', chemins['donnees_modele'])
    return chemins


def test_fenetre_sans_lot_complete_par_les_donnees_origine():
    fenetre, sources = construire_fenetre([], ORIGINE, 200)

    assert len(fenetre) == 200
    assert sources == ['data/carprice.csv (200 véhicules)']


def test_fenetre_avec_peu_de_lots(tmp_path):
    lots = creer_lots(tmp_path / 'lots', [1, 2])

    fenetre, sources = construire_fenetre(lots, ORIGINE, 200)

    assert len(fenetre) == 200
    assert sources[0] == 'data/carprice.csv (80 véhicules)'
    assert len(sources) == 3
    # le dernier lot est à la fin de la fenêtre
    dernier = pd.read_csv(lots[-1])
    assert fenetre[CIBLE].tail(60).tolist() == dernier['price'].tolist()


def test_fenetre_glissante(tmp_path):
    lots = creer_lots(tmp_path / 'lots', [1, 2, 3, 4, 5])

    fenetre, sources = construire_fenetre(lots, ORIGINE, 200)

    assert len(fenetre) == 240
    assert [Path(source).name for source in sources] == [lot.name for lot in lots[1:]]


def test_nouveau_modele_mis_en_service(tmp_path, modele_en_service):
    lots = creer_lots(tmp_path / 'lots', [1, 2, 3, 4])
    ancien_modele = modele_en_service['modele'].read_bytes()

    resultat = reentrainer(lots, charger_seuils(), **modele_en_service)

    assert resultat['remplace']
    assert resultat['challenger_dernier_lot']['mae'] < resultat['champion_dernier_lot']['mae']
    # l'ancien modèle est archivé
    archive = modele_en_service['archives'] / lots[-1].stem
    assert (archive / 'model.pkl').read_bytes() == ancien_modele
    assert (archive / 'reference.json').exists() and (archive / 'donnees_modele.csv').exists()
    # le nouveau modèle, ses données et sa référence sont en place
    assert modele_en_service['modele'].read_bytes() != ancien_modele
    donnees = pd.read_csv(modele_en_service['donnees_modele'])
    assert 'tesla' in set(donnees['marque'])
    reference = charger_reference(modele_en_service['reference'])
    assert 'tesla' in reference['valeurs_connues']['marque']
    assert reference['performance'] == resultat['challenger_fenetre']
    assert reference['nb_vehicules'] == len(donnees)


def test_modele_en_service_conserve_s_il_est_meilleur(tmp_path, modele_en_service):
    lots = creer_lots(tmp_path / 'lots', [1, 2, 3, 4])
    # modèle en service entraîné sur la fenêtre complète, dernier lot compris : il connaît déjà ce lot
    fenetre, _ = construire_fenetre(lots, ORIGINE, 200)
    sauvegarder(creer_modele().fit(fenetre.drop(CIBLE, axis=1), fenetre[CIBLE]), modele_en_service['modele'])
    avant = {nom: chemin.read_bytes() for nom, chemin in modele_en_service.items() if nom != 'archives'}

    resultat = reentrainer(lots, charger_seuils(), **modele_en_service)

    assert not resultat['remplace']
    assert not modele_en_service['archives'].exists()
    for nom, contenu in avant.items():
        assert modele_en_service[nom].read_bytes() == contenu


def test_rapport(tmp_path, modele_en_service):
    lots = creer_lots(tmp_path / 'lots', [1, 2, 3, 4])

    resultat = reentrainer(lots, charger_seuils(), **modele_en_service)
    rapport = rapport_markdown(resultat)

    assert '**Décision : nouveau modèle mis en service**' in rapport
    assert lots[-1].name in rapport


def test_modele_reentraine_utilisable(tmp_path, modele_en_service):
    lots = creer_lots(tmp_path / 'lots', [1, 2, 3, 4])
    reentrainer(lots, charger_seuils(), **modele_en_service)

    with open(modele_en_service['modele'], 'rb') as file:
        model = pickle.load(file)
    donnees = pd.read_csv(modele_en_service['donnees_modele'])

    assert len(model.predict(donnees.drop(CIBLE, axis=1).head(5))) == 5
