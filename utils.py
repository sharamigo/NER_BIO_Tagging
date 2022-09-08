# -*- coding: utf-8 -*-
import pandas as pd

import csv

import os
dirname = os.path.dirname(__file__)

# read the CSV-files
def readCSVFile(filename):
    df = pd.read_csv(os.path.join(dirname, filename), encoding = "UTF-8")
    return df

def createCSVFile(inputData):
    fields = ["Lfd-No.", "Sentence", "Word", "Category", "Tag"]
    rows = inputData
    
    filename = "./output/ner_dataset.csv"   
    with open(filename, 'w+', encoding='utf-8', newline='') as csvfile:
        csvwriter = csv.writer(csvfile) 
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)
    if os.path.exists(filename):
        print("The file " + filename + " has been generated successfully")