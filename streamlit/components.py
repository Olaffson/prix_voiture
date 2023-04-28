import pandas as pd
import pickle

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
    
    # récupération de 'model' avec pikle
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)

    # prédiction
    y_pred = model.predict(X)

    return y_pred
