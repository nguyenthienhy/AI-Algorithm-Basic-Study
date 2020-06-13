import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
import string

def process_document(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

stop_words = stopwords.words('english')

def remove_stopwords(text):
    words = [w for w in text if w not in stop_words]
    return words

def combine_text(list_of_text):
    combined_text = ' '.join(list_of_text)
    return combined_text
def word_2_vector(w2vec_model , text):
    doc = [word for word in text if word in w2vec_model.vocab]
    return np.mean(w2vec_model[doc], axis=0)




