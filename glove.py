import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

# Chemin vers le fichier GloVe
path_to_glove_file = 'C:/Users/hcroc/OneDrive/Bureau/glove.6B/glove.6B.300d.txt'

# Dictionnaire pour stocker les vecteurs GloVe  (mot -> vecteur)    
word2vec = {}

# Ouvrir le fichier GloVe
with open(path_to_glove_file, encoding="utf8") as f:
    # Parcourir toutes les lignes du fichier
    for line in f:
        # Séparer la ligne en mots
        values = line.split()
        # Le premier élément est le mot
        word = values[0]
        # Les éléments restants sont les composantes du vecteur
        vec = np.asarray(values[1:], dtype='float32')
        # Ajouter le mot et le vecteur au dictionnaire
        word2vec[word] = vec
        
def find_closest_embeddings(embedding):
    return sorted(word2vec.keys(), key=lambda word: spatial.distance.euclidean(word2vec[word], embedding))

def cosine_similarity_words(word1, word2):
    vec1 = word2vec[word1]
    vec2 = word2vec[word2]
    similarity = cosine_similarity([vec1], [vec2])
    return similarity[0][0]

trust_words = {
    "yeah": word2vec["yeah"],
    "ok": word2vec["ok"],
    "yes": word2vec["yes"],
    "please": word2vec["please"],
    "right": word2vec["right"],
    "should": word2vec["should"],
    "dear": word2vec["dear"],
    "may": word2vec["may"],
    "welcome": word2vec["welcome"],
    "indeed": word2vec["indeed"],
    "facts": word2vec["facts"],
    "good": word2vec["good"],
    "hi": word2vec["hi"],
    "happy": word2vec["happy"],
    "me": word2vec["me"],
    "we": word2vec["we"],
    "excuse": word2vec["excuse"],
    "thank": word2vec["thank"],
    "agree": word2vec["agree"],
    "course": word2vec["course"],
}

mistrust_words = {
    "not" : word2vec["not"],
    "hate": word2vec["hate"],
    "repeat": word2vec["repeat"],
    "hurry": word2vec["hurry"],
    "slow": word2vec["slow"],
    "what": word2vec["what"],
    "doubt": word2vec["doubt"],
    "mistrust": word2vec["mistrust"],
    "disrespect": word2vec["disrespect"],
    "question": word2vec["question"],
    "suspicion": word2vec["suspicion"],
    "unconvinced": word2vec["unconvinced"],  
    "disagree": word2vec["disagree"],
}

def find_similar_words(sentence, threshold=0.6):
    trust_words_counter = 0
    for word in sentence.lower().split():
        if word in word2vec:
            for trust_word, trust_vector in trust_words.items():
                similarity = cosine_similarity([word2vec[word]], [trust_vector])
                if similarity > threshold:
                    #similar_words.append(word)
                    trust_words_counter+=1
                    break
    #return similar_words
    return trust_words_counter

def find_nonsimilar_words(sentence, threshold=0.6):
    mistrust_words_counter = 0
    for word in sentence.lower().split():
        if word in word2vec:
            for mistrust_word, mistrust_vector in mistrust_words.items():
                similarity = cosine_similarity([word2vec[word]], [mistrust_vector])
                if similarity > threshold:
                    #similar_words.append(word)
                    mistrust_words_counter+=1
                    break
    #return similar_words
    return mistrust_words_counter


#sentence = "Please be happy and welcome"
#counter = find_nonsimilar_words(sentence)
#print(counter)

"""
print(find_closest_embeddings(word2vec["king"])[1:6])

print(find_closest_embeddings(
    word2vec["twig"] - word2vec["branch"] + word2vec["hand"]
)[:5])
"""
#print(cosine_similarity_words("pleased", "please"))
#print(cosine_similarity_words("man", "woman"))





