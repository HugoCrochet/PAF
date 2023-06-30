import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.svm import SVC
import pympi

def somme_couples(liste_couples):
    somme = [0, 0, 0]  # Initialisation du couple de somme à [0, 0, 0]
    for couple in liste_couples:
        somme[0] += couple[0]  # Somme des premières valeurs des couples
        somme[1] += couple[1]  # Somme des deuxièmes valeurs des couples
        somme[2] += couple[2]  # Somme des troisièmes valeurs des couples
    return tuple(somme)  # Conversion de la liste en un couple de valeurs
        
def extract_data():
    X=[]
    Y=[]
    for i in [9,10,12,15,18,19,24,26,27,30]:
        
        file ="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/"+str(i)+"/"+str(i)+".eaf"
        eaf = pympi.Elan.Eaf(file)
        annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
        
        earlyfusion="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/data_earlyfusion/"+str(i)
        data_sousdossier=[]
        
        for num_segment in range(len(os.listdir(earlyfusion))): #on parcourt tous les segments du sous-dossier
            
            file_name=earlyfusion+"/segment_"+str(num_segment)+".npy" #nom du segment dans notre dossier
            data_segment_str=list(np.load(file_name)) #on charge le segment dans la variable data_segment_str
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

    for j in range(0,10):
        for i in range(j,j+1):
            X_test=X[i]
            Y_test=Y[i]
            X_train=[]
            Y_train=[]
            for j in range(0,10):
                if j!=i:
                    X_train+= X[j]
                    Y_train+= Y[j] 
            
            model = RandomForestClassifier()
            param_grid = {
                'n_estimators': [100, 200, 300, 400, 350, 150, 500],
                'max_depth': [None, 5, 10, 15, 20, 25, 30, 35, 40],
                'min_samples_leaf': [1, 2, 4, 8, 16, 32, 64, 128, 256]
            }

            if cross_validation: 
                grid_search = GridSearchCV(model, param_grid, cv=5)
                grid_search.fit(X_train, Y_train)
                best_params = grid_search.best_params_
                print(best_params)
            else:
                model = RandomForestClassifier(max_depth=15, min_samples_leaf=16, n_estimators=500)
                model.fit(X_train, Y_train)
                predictions = model.predict(X_test)
                accuracy = accuracy_score(Y_test, predictions)
                recall = recall_score(Y_test, predictions, average='weighted')
                f1 = f1_score(Y_test, predictions, average='weighted')
                print("Accuracy:", accuracy)
                print("Recall:", recall)
                print("F1 Score:", f1)

def regression(X, Y, cross_validation=False):
    for i in range(7, 8):
        X_test = X[i]
        Y_test = Y[i]
        X_train = []
        Y_train = []
        for j in range(0, 10):
            if j != i:
                X_train += X[j]
                Y_train += Y[j]
        
        model = LinearRegression()
        
        if cross_validation:
            # Paramètres pour la recherche en grille
            param_grid = {
                'fit_intercept': [True, False],
                'normalize': [True, False]
            }
            
            # Recherche en grille avec validation croisée
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            print(best_params)
        else:
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(Y_test, predictions)
            score = model.score(X_test, Y_test)
            print(mse)
            print(score)
            print(comparer_elements(predictions, Y_test))
            
def vector(X, Y, cross_validation=False):
    for i in range(7, 8):
        X_test = X[i]
        Y_test = Y[i]
        X_train = []
        Y_train = []
        for j in range(0, 10):
            if j != i:
                X_train += X[j]
                Y_train += Y[j]

        model = SVC()

        if cross_validation:
            # Paramètres pour la recherche en grille
            param_grid = {
                'kernel': ['linear', 'rbf'],
                'C': [0.1, 1, 10],
                'epsilon': [0.1, 0.01, 0.001]
            }

            # Recherche en grille avec validation croisée
            grid_search = GridSearchCV(model, param_grid, cv=5)
            grid_search.fit(X_train, Y_train)
            best_params = grid_search.best_params_
            print(best_params)
        else:
            model.fit(X_train, Y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(Y_test, predictions)
            score = model.score(X_test, Y_test)
            print(mse)
            print(score)
            print(comparer_elements(predictions, Y_test))



X, Y = extract_data()
learn(X, Y, cross_validation=False)
#regression(X, Y, cross_validation=False)
#vector(X, Y, cross_validation=False)

