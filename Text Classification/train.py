import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer , CountVectorizer
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix , classification_report
from sklearn.naive_bayes import *
from nltk.corpus import stopwords

# đọc dữ liệu
D_data = pd.read_csv("Data_train/data_bbc.csv")
X_data = D_data["content"]
y_data = D_data["category"]

X_data , y_data = shuffle(X_data , y_data , random_state=42)

# chia 60 train , 20 validation
X_data_train_and_validation, X_T_data, y_data_train_and_validation, y_T_data = train_test_split(X_data, y_data, 
													test_size = 0.2, random_state=42)

X_data_train_and_validation , y_data_train_and_validation = shuffle(X_data_train_and_validation , 
                                                                    y_data_train_and_validation
                                                                    , random_state=42)

X_T_data , y_T_data = shuffle(X_T_data , y_T_data , random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1 , 2), min_df = 10 , max_df = 1. , stop_words=stopwords.words('english') , max_features = 300)

X_data_train_and_validation = vectorizer.fit_transform(X_data_train_and_validation).toarray()

X_train , X_val , y_train , y_val = train_test_split(X_data_train_and_validation, y_data_train_and_validation, 
                                                    test_size = 0.2, random_state=42)


clf = MultinomialNB()

clf.fit(X_train , y_train)

print("Training accuracy : " + str(clf.score(X_train , y_train)) + "\n")

y_predict = clf.predict(X_val)

print("Validation report : ")
print(classification_report(y_val , y_predict))

def predictTest(filename):
    f = open(filename , encoding="utf8")
    content = f.readlines()
    string = ""
    X_test = []
    for cnt in content:
        string = string + " " + cnt
    X_test.append(string)
    X_test = vectorizer.transform(X_test).toarray()
    return clf.predict(X_test)

def report(X_T , y_T):
	X_T = vectorizer.transform(X_T)
	y_predict = clf.predict(X_T)
	print(classification_report(y_T , y_predict))

print("Testing report : ")
report(X_T_data , y_T_data)

print("-----Tesing a example-----")
print(predictTest("Data_test/test_sports.txt"))
