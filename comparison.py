import speech_recognition as sr

# Transcription de l'enregistrement audio en texte
sourceStr = "test.wav"
r = sr.Recognizer()
with sr.AudioFile(sourceStr) as source:
    audio_data = r.record(source)
    text = r.recognize_google(audio_data, language="en-US")  # Utilisez la langue appropriée si nécessaire

# Comparaison avec les mots de la liste
word_list = ["should", "excuse", "like"]
for word in word_list:
    if word in text:
        print(f"Le mot '{word}' est prononcé dans l'enregistrement.")
    else:
        print(f"Le mot '{word}' n'est pas prononcé dans l'enregistrement.")
        
        
        

        
