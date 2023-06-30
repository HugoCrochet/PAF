from itertools import combinations

def partitionner_liste(liste):
    partitions = []
    for r in range(2, len(liste) + 1):
        for comb in combinations(liste, r):
            partitions.append(list(comb))
    return partitions


def check_repetition(phrase):
    mots = phrase.lower().split()  # Mise en minuscule des mots de la phrase
    n = len(mots)
    is_repetition = 0
    if n < 2:
        return is_repetition
    if n == 2:
        if mots[0] == mots[1]:
            is_repetition = 1
        return is_repetition
    if n == 3:
        if mots[0] == mots[1] or mots[1] == mots[2] or mots[0] == mots[2]:
            is_repetition = 1
        return is_repetition
    if n > 3:
        moitie = n // 2  # Trouver l'indice de la moitié de la liste
        sous_liste1 = mots[:moitie]  # Extraire les éléments jusqu'à la moitié
        sous_liste2 = mots[moitie:]
        partition1 = partitionner_liste(sous_liste1)
        partition2 = partitionner_liste(sous_liste2)
        
        for sublist1 in partition1:
            if sublist1 in partition2:
                is_repetition = 1
                break
        
        return is_repetition
    

