import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
import pympi

def somme_couples(liste_couples):
    somme = [0, 0, 0]  # Initialisation du couple de somme à [0, 0, 0]
    for couple in liste_couples:
        somme[0] += couple[0]  # Somme des premières valeurs des couples
        somme[1] += couple[1]  # Somme des deuxièmes valeurs des couples
        somme[2] += couple[2]  # Somme des troisièmes valeurs des couples
    return tuple(somme)  # Conversion de la liste en un couple de valeurs
        
def extract_data(nom_sousdossier):
    X=[]
    Y=[]
    for i in [9,10,12,15,18,19,24,26,27,30]:
        
        file ="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/"+str(i)+"/"+str(i)+".eaf"
        eaf = pympi.Elan.Eaf(file)
        annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
        
        sousdossier="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/data_"+nom_sousdossier+"/"+str(i)
        data_sousdossier=[]
        
        for num_segment in range(len(os.listdir(sousdossier))): #on parcourt tous les segments du sous-dossier
            
            file_name=sousdossier+"/segment_"+str(num_segment)+".npy" #nom du segment dans notre dossier
            data_segment_str= np.load(file_name) #on charge le segment dans la variable data_segment_str
            data_segment_str = data_segment_str.ravel()
            data_segment=[float(x) for x in data_segment_str[:]] #on convertit les valeurs en float
            data_sousdossier.append(data_segment) #on ajoute le segment à la liste des segments du sous-dossier
 
        ensemble_labels_sousdossier=[]

        for k in range(len(data_sousdossier)):

            tag =annots[k][2]  #on récupère le tag du segment
                
            if tag=="Neutral":
                label_segment=[0,1,0]
            if tag=="Trusting":
                label_segment=[0,0,1]
            if tag=="Mistrusting":
                label_segment=[1,0,0]
                
            ensemble_labels_sousdossier.append(label_segment)

        X.append(data_sousdossier)
        Y.append(ensemble_labels_sousdossier)

    return(X,Y)

def comparer_elements(X, Y): 
    res=0
    for element_x , elm_Y in zip(X,Y):
        couple_converti = list([1 if val == max(element_x) else 0 for val in element_x])
        if couple_converti == elm_Y:
            res+=1
        
    return res/len(Y) #on retourne le pourcentage de bonnes prédictions


def learn(X,Y,cross_validation=False):

    for i in range(9,10):
        X_test=X[i]
        Y_test=Y[i]
        X_train=[]
        Y_train=[]
        for j in range(0,10): # ajout de toutes les données sauf celles de l'interaction choisie pour le test
            if j!=i:
                X_train+= X[j]
                Y_train+= Y[j] 
        
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [100, 200, 300, 400, 350, 150, 500],
            'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 128, 256]
        }

        if cross_validation: #si je veux faire la cross validation
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            print(best_params)
        else:
            model = RandomForestClassifier(max_depth=15, min_samples_leaf=16, n_estimators=500)
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            #accuracy = accuracy_score(Y_test, predictions)
            #score = model.score(X_test, Y_test)
            #print(accuracy)
            #print(score)
            
    return predictions

X_aus, Y_aus = extract_data("aus")
X_distance, Y_distance = extract_data("distance")
X_emotion, Y_emotion = extract_data("emotion")
X_geste, Y_geste = extract_data("geste")
X_mouvement, Y_mouvement = extract_data("mouvement")
X_prosodie, Y_prosodie = extract_data("prosodie")
X_semantique, Y_semantique = extract_data("semantique")

X_earlyfusion, Y_earlyfusion = extract_data("earlyfusion")

predictions_aus = learn(X_aus, Y_aus, cross_validation=False)
predictions_distance = learn(X_distance, Y_distance, cross_validation=False)
predictions_emotion = learn(X_emotion, Y_emotion, cross_validation=False)
predictions_geste = learn(X_geste, Y_geste, cross_validation=False)
predictions_mouvement = learn(X_mouvement, Y_mouvement, cross_validation=False)
predictions_prosodie = learn(X_prosodie, Y_prosodie, cross_validation=False)
predictions_semantique = learn(X_semantique, Y_semantique, cross_validation=False)

predictions = []
label_somme = [0,0,0]
label_after_vote = [0,0,0]

for i in range(len(predictions_aus)):
    for k in range(len(predictions_aus[i])):
        label_somme[k] += predictions_aus[i][k] + predictions_distance[i][k] + predictions_emotion[i][k] + predictions_geste[i][k] + predictions_mouvement[i][k] + predictions_prosodie[i][k] + predictions_semantique[i][k]
    a = label_somme[0]
    b = label_somme[1]
    c = label_somme[2]
    if a > b and a > c:
        label_after_vote = [1, 0, 0]
    elif b > a and b > c:
        label_after_vote = [0, 1, 0]
    elif c > a and c > b:
        label_after_vote = [0, 0, 1]
    elif a == b and b == c:
        label_after_vote = [0, 1, 0]
    elif a == b and b > c:
        label_after_vote = [1, 0, 0]
    elif c == b and b > a:
        label_after_vote = [0, 0, 1]
    elif c == a and a > b:
        label_after_vote = [0, 1, 0]
    else:
        label_after_vote = [0, 1, 0]
    predictions.append(label_after_vote)
    label_somme = [0,0,0]
    label_after_vote = [0,0,0]
    
print(comparer_elements(predictions, Y_earlyfusion[9]))
    

