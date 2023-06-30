import speech_recognition as sr
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def speech_to_text_with_timestamps(num_dossier, num_person, start_time_ms, end_time_ms):
    
    audio_file = "C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/"+num_dossier+"/person"+num_person+".wav"
    
    # Conversion des timestamps en secondes
    start_time = start_time_ms / 1000
    end_time = end_time_ms / 1000

    # Initialiser le recognizer
    r = sr.Recognizer()

    # Fonction pour découper l'audio et effectuer la reconnaissance vocale
    def process_audio_segment(audio_segment):
        # Effectuer la reconnaissance vocale sur le segment
        try:
            text = r.recognize_google(audio_segment, language="en-US")
            return text
        except sr.UnknownValueError:
            return ""

    # Ouvrir le fichier audio en tant qu'objet AudioFile
    with sr.AudioFile(audio_file) as source:
        # Diviser l'enregistrement audio en segments
        audio_duration = source.DURATION

        # Vérifier que les timestamps sont valides
        if start_time < 0 or end_time > audio_duration or start_time >= end_time:
            print("Invalid timestamps.")
            return None

        # Extraire le segment audio
        audio_segment = r.record(source, offset=start_time, duration=end_time-start_time)

        # Effectuer la reconnaissance vocale sur le segment
        segment_text = process_audio_segment(audio_segment)

        # Appliquer le traitement des mots vides (stop words)
        words = word_tokenize(segment_text)
        filtered_words = [word for word in words if word.casefold() not in stop_words]
        filtered_sentence = ' '.join(filtered_words)

        return filtered_sentence


# Utilisation de la fonction
audio_file = "C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/10/person1.wav"
