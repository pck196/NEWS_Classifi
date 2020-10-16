# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import nltk
import unicodedata
import re
nltk.download('stopwords')
nltk.download('punkt')
import contractions
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

dataset = pd.read_csv('news-data.csv')

# build train and test datasets
txtdt = dataset['text'].values
category = dataset['category'].values

train_txtdt = txtdt[:1560]
train_category = category[:1560]

test_txtdt = txtdt[1560:]
test_category = category[1560:]

# preprocess document
ps = nltk.porter.PorterStemmer()
stop_words = nltk.corpus.stopwords.words('english')

def expand_contractions(text):
    return contractions.fix(text)


def simple_stemming(text, stemmer=ps):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]
    
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text

def get_metrics(true_labels, predicted_labels):
    
    print('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels, 
                                               predicted_labels),
                        4))
    print('Precision:', np.round(
                        metrics.precision_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('Recall:', np.round(
                        metrics.recall_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))
    print('F1 Score:', np.round(
                        metrics.f1_score(true_labels, 
                                               predicted_labels,
                                               average='weighted'),
                        4))

def pre_process_document(document):
    
    # lower case
    document = document.lower()

    # remove accented characters
    document = remove_accented_chars(document)
    
    # expand contractions    
    document = expand_contractions(document)

    # remove special characters and\or digits
    # insert spaces between special characters to isolate them
    special_char_pattern = re.compile(r'([{.(-)!}])')
    document = special_char_pattern.sub(" \\1 ", document)
    document = remove_special_characters(document, remove_digits=True)  

    # stemming text
    document = simple_stemming(document)      
    
    # remove stopwords
    document = remove_stopwords(document, is_lower_case=True, stopwords=stop_words)

    # remove extra whitespace
    document = re.sub(' +', ' ', document)
    document = document.strip()
        
    
    return document


pre_process = np.vectorize(pre_process_document)

# normalizing data
norm_train_txtdt = pre_process(train_txtdt)
norm_test_txtdt = pre_process(test_txtdt)

# build TFIDF features
tv = TfidfVectorizer(use_idf=True, min_df=5, max_df=1.0, ngram_range=(1,2),
                     sublinear_tf=True)

# train model
tv_train_features = tv.fit_transform(norm_train_txtdt)
tv_test_features = tv.transform(norm_test_txtdt)

# fit model
svc=SVC(random_state=41)
svc.fit(tv_train_features, train_category)

f = open('svc.pickle', 'wb')
pickle.dump(svc, f)
f.close()

f = open('tv.pickle', 'wb')
pickle.dump(tv, f)
f.close()


# predict on test data
svc_tfidf_predictions = svc.predict(tv_test_features)
print(get_metrics(true_labels=test_category, predicted_labels=svc_tfidf_predictions))