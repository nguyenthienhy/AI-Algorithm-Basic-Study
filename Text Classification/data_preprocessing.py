import numpy as np
import nltk
import process_data
import pandas as pd
import os
import re
import nltk
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

PARAGRAPH = []
LABEL = []


def text_preprocessing(X):
    documents = []
    stemmer = WordNetLemmatizer()
    for sen in range(0, len(X)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(X[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)

        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document)

        # Substituting multiple spaces with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)

        # Removing prefixed 'b'
        document = re.sub(r'^b\s+', '', document)

        # Converting to Lowercase
        document = document.lower()

        # Lemmatization
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
    return documents


def searchFile(basepath, category):
    for entry in os.listdir(basepath):
        if os.path.isfile((os.path.join(basepath, entry))):
            content = process_data.readData(basepath + '/' + entry)
            content = process_data.remove_N(content)
            content = process_data.remove_mark_all_sentences(content)
            content = process_data.convert_to_lower_case(content)
            content = text_preprocessing(content)
            dataInline = ""
            for cnt in content:
                dataInline = dataInline + " " + cnt
            if dataInline != "":
                PARAGRAPH.append(dataInline)
                if category == "Economy":
                    LABEL.append(1)
                elif category == "Entertainment":
                    LABEL.append(2)
                elif category == "Politics":
                    LABEL.append(3)
                elif category == "Sports":
                    LABEL.append(4)
                elif category == "Technology":
                    LABEL.append(5)


searchFile("Data_Raw/business", "Economy")
searchFile("Data_Raw/entertainment", "Entertainment")
searchFile("Data_Raw/politics", "Politics")
searchFile("Data_Raw/sport", "Sports")
searchFile("Data_Raw/tech", "Technology")

df = pd.DataFrame()
df['content'] = PARAGRAPH
df['category'] = LABEL
df.to_csv("data.csv", encoding="utf-8")

'''
def searchDir(basepath):
    for entry in os.listdir(basepath):
        basepath_sub = basepath
        if os.path.isdir((os.path.join(basepath_sub, entry))):
            basepath_sub = basepath + '/' + entry
            searchFile(basepath_sub)
            searchDir(basepath_sub)
'''
# searchDir(basepath)

#
