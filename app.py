import streamlit as st
import pandas as pd

df = pd.read_csv('voiture.csv')

st.title('Affichage du df')
st.dataframe(df)


st.title("Sélection des paramétres du véhicule pour l'estimation du prix")
col1, col2, col3, col4 = st.columns(4)

marques = list(df['marque'].unique())
selected_marque = col1.selectbox("Sélectionnez la marque :", options=marques)

carburants = list(df['carburant'].unique())
selected_carburant = col1.selectbox("Sélectionnez le carburant :", options=carburants)

turbos = list(df['turbo'].unique())
selected_turbo = col1.selectbox("Sélectionnez le turbo :", options=turbos)

nombre_portess = list(df['nombre_portes'].unique())
selected_nombre_portes = col1.selectbox("Sélectionnez le nombre de portes :", options=nombre_portess)

risques = list(df['risque_assurance'].unique())
selected_risque = col1.selectbox("Sélectionnez le risque d'assurance :", options=risques)

courses = list(df['course'].unique())
selected_course = col1.selectbox("Sélectionnez la course :", options=courses)

taux_compressions = list(df['taux_compression'].unique())
selected_taux_compression = col1.selectbox("Sélectionnez le taux de compression :", options=taux_compressions)

############################################################################################################################

modeles = list(df['modele'].unique())
selected_modele = col2.selectbox("Sélectionnez le modele :", options=modeles)

consommation_villes = list(df['consommation_ville'].unique())
selected_consommation_ville = col2.selectbox("Sélectionnez la consommation en ville :", options=consommation_villes)

consommation_autoroutes = list(df['consommation_autoroute'].unique())
selected_consommation_autoroute = col2.selectbox("Sélectionnez la consommation sur autoroute :", options=consommation_autoroutes)

type_vehiculess = list(df['type_vehicule'].unique())
selected_type_vehicule = col2.selectbox("Sélectionnez le type de véhicule :", options=type_vehiculess)

roues_motricess = list(df['roues_motrices'].unique())
selected_roues_motrices = col2.selectbox("Sélectionnez les roues_motrices :", options=roues_motricess)

emplacement_moteurs = list(df['emplacement_moteur'].unique())
selected_emplacement_moteur = col2.selectbox("Sélectionnez l'emplacement du moteur :", options=emplacement_moteurs)


############################################################################################################################

longueur_voitures = list(df['longueur_voiture'].unique())
selected_longueur_voiture = col3.selectbox("Sélectionnez la longueur de la voiture :", options=longueur_voitures)

largeur_voitures = list(df['largeur_voiture'].unique())
selected_largeur_voiture = col3.selectbox("Sélectionnez la largeur de la voiture :", options=largeur_voitures)

hauteur_voitures = list(df['hauteur_voiture'].unique())
selected_hauteur_voiture = col3.selectbox("Sélectionnez la hauteur de la voiture :", options=hauteur_voitures)

poids_voitures = list(df['poids_voiture'].unique())
selected_poids_voiture = col3.selectbox("Sélectionnez le poids de la voiture :", options=poids_voitures)

puissances = list(df['puissance'].unique())
selected_puissance = col3.selectbox("Sélectionnez la puissance :", options=puissances)

empattements = list(df['empattement'].unique())
selected_empattement = col3.selectbox("Sélectionnez l'empattement :", options=empattements)

############################################################################################################################

type_moteurs = list(df['type_moteur'].unique())
selected_type_moteurs = col4.selectbox("Sélectionnez le type de moteur :", options=type_moteurs)

nombre_cylindress = list(df['nombre_cylindres'].unique())
selected_nombre_cylindres = col4.selectbox("Sélectionnez le nombre de cylindres :", options=nombre_cylindress)

taille_moteurs = list(df['taille_moteur'].unique())
selected_taille_moteur = col4.selectbox("Sélectionnez la taille du moteur :", options=taille_moteurs)

systeme_carburants = list(df['systeme_carburant'].unique())
selected_systeme_carburant = col4.selectbox("Sélectionnez le systeme pour le carburant :", options=systeme_carburants)

taux_alesages = list(df['taux_alesage'].unique())
selected_taux_alesage = col4.selectbox("Sélectionnez le taux d'alésage :", options=taux_alesages)

tour_moteurs = list(df['tour_moteur'].unique())
selected_tour_moteur = col4.selectbox("Sélectionnez les tours moteur :", options=tour_moteurs)

