import pympi
import numpy as np
from comparison2 import speech_to_text_with_timestamps
from glove import find_similar_words
from glove import find_nonsimilar_words
from laughter import check_laughter
from repetition import check_repetition

for i in [15]:
    file ="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/"+str(i)+"/"+str(i)+".eaf"
    eaf = pympi.Elan.Eaf(file)
    annots = sorted(eaf.get_annotation_data_for_tier('Trust'))
    num_segment=0
    for segment in annots:
        saved_file="C:/Users/hcroc/OneDrive/Bureau/PAF_2023/data_semantique/"+str(i)+"/segment_"+str(num_segment)
        #print("begin : "+str(segment[0])+" end : "+str(segment[1]))
        
        sentence_over_segment1 = speech_to_text_with_timestamps(str(i), "1", segment[0], segment[1])
        sentence_over_segment2 = speech_to_text_with_timestamps(str(i), "2", segment[0], segment[1])
        
        print("segment : "+str(num_segment))
        print("debut : "+str(segment[0])+" fin : "+str(segment[1]))
        print("speech1 : " + sentence_over_segment1)
        print("speech2 : " + sentence_over_segment2)
        
        print("trust1 : ")
        trust_count1 = find_similar_words(sentence_over_segment1)
        print("mistrust1 : ")
        mistrust_count1 = find_nonsimilar_words(sentence_over_segment1)
        print("repetition1 : ")
        repetition1 = check_repetition(sentence_over_segment1)
        print("rire1 : ")
        rire1 = check_laughter(segment[0], segment[1], str(i), "1")
        
        print("trust2 : ")
        trust_count2 = find_similar_words(sentence_over_segment2)
        print("mistrust2 : ")
        mistrust_count2 = find_nonsimilar_words(sentence_over_segment2)
        print("repetition2 : ")
        repetition2 = check_repetition(sentence_over_segment2)
        print("rire2 : ")
        rire2 = check_laughter(segment[0], segment[1], str(i), "2")
        
        data = [trust_count1, mistrust_count1, repetition1, rire1, trust_count2, mistrust_count2, repetition2, rire2]
        data=np.save(saved_file,np.array(data))
        print(np.load(saved_file+".npy"))
        
        num_segment+=1