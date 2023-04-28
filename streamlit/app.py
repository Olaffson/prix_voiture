import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import make_column_selector
from components import predict_price

def main():
    """
    Application Streamlit pour donner une estimation de prix à un véhicule.
    """
    st.set_page_config(page_title="Estimator", page_icon="chart_with_upwards_trend", layout="wide")
    # page_icon=":movie_camera:",

    st.title("Estimation du prix du véhicule")
    
    df = pd.read_csv('../data/data_utilisable.csv')

    # Organiser les box de selections dans des colonnes
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])

    # colonne 1
    marques = list(df['marque'].unique())
    selected_marque = col1.selectbox("Sélectionnez la marque :", options=marques)
    modeles = list(df['modele'].unique())
    selected_modele = col1.selectbox("Sélectionnez le modele :", options=modeles)
    type_vehiculess = list(df['type_vehicule'].unique())
    selected_type_vehicule = col1.selectbox("Sélectionnez le type de véhicule :", options=type_vehiculess)
    carburants = list(df['carburant'].unique())
    selected_carburant = col1.selectbox("Sélectionnez le carburant :", options=carburants)
    nombre_portess = list(df['nombre_portes'].unique())
    selected_nombre_portes = col1.selectbox("Sélectionnez le nombre de portes :", options=nombre_portess)

    # colonne 2
    turbos = list(df['turbo'].unique())
    selected_turbo = col2.selectbox("Sélectionnez le turbo :", options=turbos)
    puissances = list(df['puissance'].unique())
    selected_puissance = col2.selectbox("Sélectionnez la puissance :", options=puissances)
    systeme_carburants = list(df['systeme_carburant'].unique())
    selected_systeme_carburant = col2.selectbox("Sélectionnez le systeme pour le carburant :", options=systeme_carburants)
    consommation_villes = list(df['consommation_ville'].unique())
    selected_consommation_ville = col2.selectbox("Sélectionnez la consommation en ville :", options=consommation_villes)
    consommation_autoroutes = list(df['consommation_autoroute'].unique())
    selected_consommation_autoroute = col2.selectbox("Sélectionnez la consommation sur autoroute :", options=consommation_autoroutes)

    # colonne 3
    longueur_voitures = list(df['longueur_voiture'].unique())
    selected_longueur_voiture = col3.selectbox("Sélectionnez la longueur de la voiture :", options=longueur_voitures)
    largeur_voitures = list(df['largeur_voiture'].unique())
    selected_largeur_voiture = col3.selectbox("Sélectionnez la largeur de la voiture :", options=largeur_voitures)
    hauteur_voitures = list(df['hauteur_voiture'].unique())
    selected_hauteur_voiture = col3.selectbox("Sélectionnez la hauteur de la voiture :", options=hauteur_voitures)
    poids_voitures = list(df['poids_voiture'].unique())
    selected_poids_voiture = col3.selectbox("Sélectionnez le poids de la voiture :", options=poids_voitures)
    empattements = list(df['empattement'].unique())
    selected_empattement = col3.selectbox("Sélectionnez l'empattement :", options=empattements)

    # colonne 4
    type_moteurs = list(df['type_moteur'].unique())
    selected_type_moteurs = col4.selectbox("Sélectionnez le type de moteur :", options=type_moteurs)
    nombre_cylindress = list(df['nombre_cylindres'].unique())
    selected_nombre_cylindres = col4.selectbox("Sélectionnez le nombre de cylindres :", options=nombre_cylindress)
    taille_moteurs = list(df['taille_moteur'].unique())
    selected_taille_moteur = col4.selectbox("Sélectionnez la taille du moteur :", options=taille_moteurs)
    tour_moteurs = list(df['tour_moteur'].unique())
    selected_tour_moteur = col4.selectbox("Sélectionnez les tours moteur :", options=tour_moteurs)
    emplacement_moteurs = list(df['emplacement_moteur'].unique())
    selected_emplacement_moteur = col4.selectbox("Sélectionnez l'emplacement du moteur :", options=emplacement_moteurs)

    # colonne 5
    roues_motricess = list(df['roues_motrices'].unique())
    selected_roues_motrices = col5.selectbox("Sélectionnez les roues_motrices :", options=roues_motricess)
    risques = list(df['risque_assurance'].unique())
    selected_risque = col5.selectbox("Sélectionnez le risque d'assurance :", options=risques)
    courses = list(df['course'].unique())
    selected_course = col5.selectbox("Sélectionnez la course :", options=courses)
    taux_compressions = list(df['taux_compression'].unique())
    selected_taux_compression = col5.selectbox("Sélectionnez le taux de compression :", options=taux_compressions)
    taux_alesages = list(df['taux_alesage'].unique())
    selected_taux_alesage = col5.selectbox("Sélectionnez le taux d'alésage :", options=taux_alesages)

    predict_button = st.button("Estimation")
    if predict_button:
        estimation = predict_price(selected_risque,
                                selected_carburant, 
                                selected_turbo, 
                                selected_nombre_portes, 
                                selected_type_vehicule, 
                                selected_roues_motrices, 
                                selected_emplacement_moteur,
                                selected_empattement, 
                                selected_longueur_voiture, 
                                selected_largeur_voiture, 
                                selected_hauteur_voiture, 
                                selected_poids_voiture, 
                                selected_type_moteurs, 
                                selected_nombre_cylindres, 
                                selected_taille_moteur, 
                                selected_systeme_carburant, 
                                selected_taux_alesage, 
                                selected_course, 
                                selected_taux_compression, 
                                selected_puissance, 
                                selected_tour_moteur,
                                selected_consommation_ville, 
                                selected_consommation_autoroute, 
                                selected_marque, 
                                selected_modele
                                )
        st.write(f"Le prix estimé pour ce véhicule est de : {estimation} €")


############################################################################################################################


#fonction de prediction du prix du véhicule
def predict_price(risque_assurance, carburant, turbo, nombre_portes, type_vehicule, roues_motrices, emplacement_moteur, empattement, longueur_voiture, largeur_voiture, hauteur_voiture, poids_voiture, type_moteur, nombre_cylindres, taille_moteur, systeme_carburant, taux_alesage, course, taux_compression, puissance, tour_moteur, consommation_ville, consommation_autoroute, marque, modele):
    """
    Estime le prix d'un véhicule à partir des caractéristiques spécifiées.

    Args:
        risque_assurance (str): le niveau de risque d'assurance du véhicule.
        carburant (str): le type de carburant utilisé par le véhicule.
        turbo (str): la présence ou non d'un turbo dans le véhicule.
        nombre_portes (int): le nombre de portes du véhicule.
        type_vehicule (str): le type de véhicule (voiture, camion, etc.).
        roues_motrices (str): le nombre de roues motrices du véhicule.
        emplacement_moteur (str): l'emplacement du moteur du véhicule (avant, arrière, etc.).
        empattement (float): la longueur entre les essieux du véhicule.
        longueur_voiture (float): la longueur du véhicule.
        largeur_voiture (float): la largeur du véhicule.
        hauteur_voiture (float): la hauteur du véhicule.
        poids_voiture (float): le poids du véhicule.
        type_moteur (str): le type de moteur du véhicule (essence, diesel, électrique, etc.).
        nombre_cylindres (int): le nombre de cylindres du moteur du véhicule.
        taille_moteur (float): la taille du moteur du véhicule.
        systeme_carburant (str): le système de carburant utilisé par le véhicule.
        taux_alesage (float): le taux d'alesage du moteur du véhicule.
        course (float): la course du moteur du véhicule.
        taux_compression (float): le taux de compression du moteur du véhicule.
        puissance (float): la puissance du moteur du véhicule.
        tour_moteur (float): la plage de régime du moteur du véhicule.
        consommation_ville (float): la consommation de carburant en ville du véhicule.
        consommation_autoroute (float): la consommation de carburant sur autoroute du véhicule.
        marque (str): la marque du véhicule.
        modele (str): le modèle du véhicule.

    Returns:
        float: la valeur estimée du véhicule en euros.
    """
    
    # Créer un DataFrame avec les valeurs sélectionnées
    X = pd.DataFrame({
        'risque_assurance' : [risque_assurance],
        'carburant' : [carburant],
        'turbo' : [turbo],
        'nombre_portes' : [nombre_portes],
        'type_vehicule' : [type_vehicule],
        'roues_motrices' : [roues_motrices],
        'emplacement_moteur' : [emplacement_moteur],
        'empattement' : [empattement],
        'longueur_voiture' : [longueur_voiture],
        'largeur_voiture' : [largeur_voiture],
        'hauteur_voiture' : [hauteur_voiture],
        'poids_voiture' : [poids_voiture],
        'type_moteur' : [type_moteur],
        'nombre_cylindres' : [nombre_cylindres],
        'taille_moteur' : [taille_moteur],
        'systeme_carburant' : [systeme_carburant],
        'taux_alesage' : [taux_alesage],
        'course' : [course],
        'taux_compression' : [taux_compression],
        'puissance' : [puissance],
        'tour_moteur' : [tour_moteur],
        'consommation_ville' : [consommation_ville],
        'consommation_autoroute' : [consommation_autoroute],
        'marque' : [marque],
        'modele' : [modele]
    })
    
    # récupération de 'model'
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # prédiction
    y_pred = model.predict(X)

    return y_pred

    
############################################################################################################################


if __name__ == "__main__":
    main()