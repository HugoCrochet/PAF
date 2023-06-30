import os
import numpy as np
import re

for i in [9,10,12,15,18,19,24,26,27,30]: 
    
    # Chemins des dossiers contenant les fichiers .npy
    semantique_folder = "data_semantique/"+str(i)+"/"
    aus_folder = "data_aus/"+str(i)+"/"
    geste_folder = "data_geste/"+str(i)+"/"
    mouvement_folder = "data_mouvement/"+str(i)+"/"
    distance_folder = "data_distance/"+str(i)+"/"
    emotion_folder = "data_emotion/"+str(i)+"/"
    prosodie_folder = "data_prosodie/"+str(i)+"/"
    earlyfusion_folder = "data_earlyfusion/"+str(i)+"/"

    # Vérifier si le dossier "data_earlyfusion" existe, sinon le créer
    if not os.path.exists(earlyfusion_folder):
        os.makedirs(earlyfusion_folder)

    # Obtenir la liste des fichiers .npy dans le dossier "data_semantique"
    #semantique_files = sorted([f for f in os.listdir(semantique_folder) if f.endswith(".npy")])
    fichiers = os.listdir(semantique_folder)
    # Filtrer les fichiers .npy et extraire les nombres des noms de fichiers
    nombres = [int(re.search(r'\d+', fichier).group()) for fichier in fichiers if fichier.endswith(".npy")]
    # Trier les fichiers .npy en fonction des nombres extraits
    semantique_files = sorted(fichiers, key=lambda fichier: int(re.search(r'\d+', fichier).group()))
    #print(semantique_files)

    # Parcourir les fichiers et les concaténer
    for i, semantique_file in enumerate(semantique_files):
        # Chemins des fichiers à concaténer
        semantique_path = os.path.join(semantique_folder, semantique_file)
        aus_path = os.path.join(aus_folder, "segment_" + str(i) + ".npy")
        geste_path = os.path.join(geste_folder, "segment_" + str(i) + ".npy")
        mouvement_path = os.path.join(mouvement_folder, "segment_" + str(i) + ".npy")
        distance_path = os.path.join(distance_folder, "segment_" + str(i) + ".npy")
        emotion_path = os.path.join(emotion_folder, "segment_" + str(i) + ".npy")
        prosodie_path = os.path.join(prosodie_folder, "segment_" + str(i) + ".npy")
        
        # Charger les fichiers .npy
        semantique_data = np.load(semantique_path)
        aus_data = np.load(aus_path)
        geste_data = np.load(geste_path)
        mouvement_data = np.load(mouvement_path)
        distance_data = np.load(distance_path)
        distance_data = distance_data.ravel() # Reshape pour avoir la même dimension que les autres données
        emotion_data = np.load(emotion_path)
        prosodie_data = np.load(prosodie_path)
        prosodie_data = prosodie_data.reshape(88,) # Reshape pour avoir la même dimension que les autres données
        
        # Concaténer les données
        earlyfusion_data = np.concatenate((aus_data, distance_data, emotion_data, geste_data, mouvement_data, prosodie_data, semantique_data))
        
        # Chemin du fichier de sortie
        earlyfusion_file = "segment_" + str(i) + ".npy"
        earlyfusion_path = os.path.join(earlyfusion_folder, earlyfusion_file)
        
        # Enregistrer les données concaténées
        np.save(earlyfusion_path, earlyfusion_data)

    

