DOSSIERS :

data_modalite : contient les 10 sous-dossiers (9,10,12,15,18,19,24,26,27,30), contenant chacun tous les vecteurs des segments des interactions.
-> se décline en aus, distance, emotion, geste, mouvement, prosodie, semantique.

data_earlyfusion : contient les 10 sous-dossiers (9,10,12,15,18,19,24,26,27,30), contenant chacun tous les vecteurs des segments des interactions. Ces derniers sont formés par concaténation des vecteurs de chaque modalité.

PAF_2023 : contient le dataset + des ressources (papiers de recherche)

réu : contient tous les fichiers texte de prise de note des différentes réunions.

---

SCRIPTS PYTHON :

comparison.py : premier script qui, a partir d'une phrase dit si des mots d'une liste se trouvent dans cette phrase.

comparison2.py : speechToText a partir d'un bout audio .wav délimité par deux time stamps, plus filtrage pour enlever les stopwords.

glove.py : charge un fichier GloVe contenant des vectorisations de mots. A partir de ces vecteurs nous effectuons des similarités cosinus pour compter le nombre de mots de "confiance" et de "méfiance" dans une phrase.

laughter.py : scrap un fichier texte pour trouver si entre deux time stamps, une personne rigole ou non.

repetition.py : detecte une éventuelle répétition dans une phrase

generationData.py : à partir des scripts ci-dessus, boucle sur tous les sous dossiers afin de constituer tous les vecteurs de la modalité "sémantique".

earlyfusion.py : en concatenant les vecteurs de toutes les modalités, forme un nouveau vecteur grace au principe d'earlyfusion.

ML.py : contient différentes fonctions pour tester des algos de machine learning pour prédire nos labels.

ML_semantique.py : random forest sur la modalité "semantique"

ML_latefusion.py : random forest sur toutes les modalités, puis fusion des resultats --> late fusion
