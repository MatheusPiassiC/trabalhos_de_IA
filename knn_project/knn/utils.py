import csv

def load_csv(path = "/../data/Iris.csv"):
    dataset = []
    with open(path,'r') as file:
        reader = csv.reader(file)

    return dataset