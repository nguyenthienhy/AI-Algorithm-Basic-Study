import pandas as pd
import process_data
import numpy as np
import gensim
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC , LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB , MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

model = gensim.models.KeyedVectors.load_word2vec_format('english.bin', binary = True)

print(model['5'])

def readData(path):
    Data = pd.read_csv(path)
    text_df = Data[["title", "company_profile", "description", "requirements", "benefits"]]
    text_df = text_df.fillna(' ')
    X = text_df[text_df.columns[0:-1]].apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
    y = Data['fraudulent']
    X = X.apply(lambda x: process_data.process_document(x))
    X = X.apply(lambda x: process_data.tokenizer.tokenize(x))
    X = X.apply(lambda x: process_data.remove_stopwords(x))
    X = X.apply(lambda x: process_data.combine_text(x))
    X = X.values
    Copus = []
    for doc in X:
        Copus.append(process_data.word_2_vector(model , doc))
    return np.array(Copus) , y

X , y = readData("fake_job_postings.csv")

X_train_0_label , X_test_0_label , y_train_0_label , y_test_0_label = train_test_split(X[y == 0] , y[y == 0]
                                                                          , test_size = 0.03 , random_state = 42)

X_train_1_label , X_test_1_label , y_train_1_label , y_test_1_label = train_test_split(X[y == 1] , y[y == 1]
                                                                                     , test_size = 0.03 , random_state = 42)

X_train = np.concatenate((X_train_0_label , X_train_1_label) , axis = 0)
X_test = np.concatenate((X_test_0_label , X_test_1_label) , axis = 0)

y_train = np.concatenate((y_train_0_label , y_train_1_label) , axis = 0)
y_test = np.concatenate((y_test_0_label , y_test_1_label) , axis = 0)

clf = MultinomialNB()

clf.fit(X_train , y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test , y_pred))

