import streamlit as st
import pandas as pd
import pickle
from sklearn.compose import make_column_selector
from components import predict_price

############################################################################################################################

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

if __name__ == "__main__":
    main()