def check_laughter(start_time_ms, end_time_ms, num_dossier, num_person):
    laughter_detected = False
    laughter_start_time = 0
    laughter_end_time = 0

    with open("C:/Users/hcroc/OneDrive/Bureau/PAF_2023/PAF_2023/Dataset/Interactions/"+num_dossier+"/GTUtterancePerson"+num_person+".txt", "r") as file:
        next(file)  # Ignorer la première ligne (en-têtes)
        lines = file.readlines()

        for line in lines:
            line = line.strip().split("\t")
            start = float(line[0])
            end = float(line[1])
            event = line[2]

            if event == "laughter":
                if start_time_ms <= start * 1000 <= end_time_ms or start_time_ms <= end * 1000 <= end_time_ms:
                    laughter_detected = True
                    laughter_start_time = start
                    laughter_end_time = end
                    break

    if laughter_detected:
        #print("Laughter detected from {} ms to {} ms.".format(laughter_start_time * 1000, laughter_end_time * 1000))
        return 1
    else:
        #print("No laughter detected in the specified time range.")
        return 0


# Utilisation du script
#start_time_ms = 82000  # Timestamp de début (en millisecondes)
#end_time_ms = 83000  # Timestamp de fin (en millisecondes)

#result = check_laughter(start_time_ms, end_time_ms, "19", "1")
#print("Result:", result)
