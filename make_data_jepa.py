import os
import csv
import random

def files_to_csv(folder_path, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if os.path.isfile(full_path):  # Check if it is a file
                random_num = random.randint(1, 100)  # Generate a random integer
                writer.writerow([full_path, random_num])

# Example usage
folder_path = '/scratch/amr10073/data/unlabeled_jepa'  # Replace with the path to your folder
output_csv = 'data_jepa.csv'  # Path to the output CSV file
files_to_csv(folder_path, output_csv)
