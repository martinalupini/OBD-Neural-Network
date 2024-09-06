import csv
import os

def add_csv_line(modello, reg, lambd, tr_acc, val_acc):
    # Apro il file in modalità 'append' per aggiungere righe senza sovrascrivere
    with open('csv_files/results.csv', mode='a', newline='') as file:
        nomi_colonne = ['modello', 'reg', 'lambda', 'tr acc', 'val acc']
        nuova_riga = {'modello': modello, 'reg': reg, 'lambda': lambd, 'tr acc': tr_acc, 'val acc': val_acc}

        writer = csv.DictWriter(file, fieldnames=nomi_colonne)

        # Scrivo l'intestazione (solo se il file è vuoto)
        if file.tell() == 0:
            writer.writeheader()

        # Aggiungo la riga passata come argomento
        writer.writerow(nuova_riga)


def clear_folder(directory):

    file_list = os.listdir(directory)

    for file_name in file_list:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
