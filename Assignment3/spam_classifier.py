# %%
import numpy as np
import pandas as pd
import chardet
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import re
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report


# %%
rawdata = open('emails.csv', 'rb').read()
result = chardet.detect(rawdata)
charenc = result['encoding']
rawdata = pd.read_csv('emails.csv', encoding=charenc, header=0, usecols=['text', 'spam'])
rawdata = rawdata.rename(columns={'spam': 'label', 'text': 'mail'})

# %%
def preprocess_text(text):
    if isinstance(text, str):
        text = re.sub(r'^Subject:', '', text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text
    else:
        return ''

# %%
### PREPROCESSING ###

data = rawdata.copy()
data['mail'] = data['mail'].apply(preprocess_text)

vectorizer_binary = CountVectorizer(binary=True, stop_words='english')
vectorizer_binary.fit(data['mail'])
binary_data_matrix = vectorizer_binary.transform(data['mail'])
binary_data_df = pd.DataFrame(binary_data_matrix.toarray(), columns=vectorizer_binary.get_feature_names_out())
binary_data_array = binary_data_df.iloc[:, :].values

vectorizer_count = CountVectorizer(binary=False, stop_words='english')
vectorizer_count.fit(data['mail'])
frequency_data_matrix = vectorizer_count.transform(data['mail'])
frequency_data_df = pd.DataFrame(frequency_data_matrix.toarray(), columns=vectorizer_count.get_feature_names_out())
frequency_data_array = frequency_data_df.iloc[:, :].values

vectorizer_tfidf = TfidfVectorizer(stop_words='english')
vectorizer_tfidf.fit(data['mail'])
tfidf_data_matrix = vectorizer_tfidf.transform(data['mail'])
tfidf_data_df = pd.DataFrame(tfidf_data_matrix.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
tfidf_data_array = tfidf_data_df.iloc[:, :].values


# %%
### TRAINING ###

# BAYESIAN NAIVE BAYES CLASSIFIER
alpha = 1 # Laplace smoothing
spam_count = data['label'].value_counts()[1]

p_cap_bnb = (spam_count+alpha)/(len(binary_data_array)+2*alpha)

p_features_cap_bnb = np.full((2, len(binary_data_array[0])), 0.0, dtype='float64')
p_features_cap_bnb[1] = (np.sum(binary_data_array[data['label'] == 1][:, 0:], axis=0, dtype='float64')+alpha)/(spam_count+alpha)
p_features_cap_bnb[0] = (np.sum(binary_data_array[data['label'] == 0][:, 0:], axis=0, dtype='float64')+alpha)/((len(binary_data_array) - spam_count)+alpha)

bias = np.log(p_cap_bnb/(1 - p_cap_bnb), dtype='float64')
weights = np.full(binary_data_array.shape[1], 0.0, dtype='float64')
for i in range(binary_data_array.shape[1]):
    bias += np.log((1-p_features_cap_bnb[1][i])/(1-p_features_cap_bnb[0][i]), dtype='float64')
    weights[i] = np.log(p_features_cap_bnb[1][i]*(1-p_features_cap_bnb[0][i])/(p_features_cap_bnb[0][i]*(1-p_features_cap_bnb[1][i])), dtype='float64')

def calc_acc_bnb(binary_data_array, labels, weights, bias):

    data_predictions = []
    for i in range(binary_data_array.shape[0]):
        if(np.dot(binary_data_array[i], weights) + bias >= 0):
            data_predictions.append(1)
        else:
            data_predictions.append(0)

    data_accuracy = np.sum(data_predictions == labels)/len(labels)
    return data_accuracy



# %%

# SVM CLASSIFIER

svm = LinearSVC(dual=True, max_iter=5000, C = 1)
svm.fit(tfidf_data_array, data['label'])
predictions_svm = svm.predict(tfidf_data_array)


# %%
# LOGISTIC REGRESSION CLASSIFIER
def sigmoid(x):
    return 1/(1 + np.exp(-x))

eta = 1e-2
w_lr = np.full(tfidf_data_array.shape[1], 0.0, dtype='float64')
iterations = 100
for i in range(iterations):
    sigmoid_array = sigmoid(np.dot(tfidf_data_array, w_lr))
    w_lr += eta * np.dot(data['label'] - sigmoid_array, tfidf_data_array)


def calc_acc_lr(frequency_data_array, labels, w_lr):

    data_predictions = []
    for i in range(frequency_data_array.shape[0]):
        if(np.dot(w_lr, frequency_data_array[i]) >= 0):
            data_predictions.append(1)
        else:
            data_predictions.append(0)
    data_accuracy = np.sum(data_predictions == labels)/len(labels)
    return data_accuracy


# %%
### TESTING ###
from scipy.stats import mode
n = 1

bnb_predictions = []
svm_predictions = []
lr_predictions = []

while True:
    file_path = f'test/email{n}.txt'
    if not os.path.exists(file_path):
        break
    with open(file_path, 'r') as file:
        mail = file.read()
        mail = preprocess_text(mail)
        mail_matrix = vectorizer_count.transform([mail])
        mail_frequency_df = pd.DataFrame(mail_matrix.toarray(), columns=vectorizer_count.get_feature_names_out())
        mail_frequency_array = mail_frequency_df.iloc[:, :].values
        mail_matrix = vectorizer_binary.transform([mail])
        mail_binary_df = pd.DataFrame(mail_matrix.toarray(), columns=vectorizer_binary.get_feature_names_out())
        mail_binary_array = mail_binary_df.iloc[:, :].values
        mail_matrix = vectorizer_tfidf.transform([mail])
        mail_tfidf_df = pd.DataFrame(mail_matrix.toarray(), columns=vectorizer_tfidf.get_feature_names_out())
        mail_tfidf_array = mail_tfidf_df.iloc[:, :].values

        bnb_prediction = 1 if np.dot(mail_binary_array, weights) + bias >= 0 else 0
        bnb_predictions.append(bnb_prediction)
        svm_prediction = svm.predict(mail_frequency_array)
        svm_predictions.append(int(svm_prediction))
        lr_prediction = 1 if np.dot(w_lr, mail_tfidf_array[0]) >= 0 else 0
        lr_predictions.append(lr_prediction)
        n += 1

classifier_predictions = [bnb_predictions, svm_predictions, lr_predictions]
prediction_matrix = np.vstack(classifier_predictions)
final_predictions = mode(prediction_matrix, axis=0, keepdims=True).mode[0]

for i in range(len(final_predictions)):
    print(final_predictions[i])

with open('predictions.txt', 'w') as output_file:
    for i, prediction in enumerate(final_predictions):
        output_file.write(f'Email {i + 1} : NB: {bnb_predictions[i]}, SVM : {svm_predictions[i]}, LR: {lr_predictions[i]},  Final prediction: {prediction}\n')

