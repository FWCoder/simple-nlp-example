# #############################################################################
# Documetation:
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
# #############################################################################

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score
import pickle

# #############################################################################
# Load some categories from the training set

input_file = "topics.csv"
output_file = "text-classifier.pkl"

nltk.download('stopwords')

df = pd.read_csv(input_file)
original_headers = list(df.columns.values)

# print(original_headers)

categories = df[original_headers[2]].unique().tolist()

print("Total dataset size: "+str(len(df)))

# Split data: 80% for training, 20% for testing
train, test = train_test_split(df, test_size=0.2)

print("Training dataset size: "+str(len(train)))
print("Test dataset size: "+str(len(test)))

train_x =  train[original_headers[1]].tolist()
train_y = train[original_headers[2]].tolist()

test_x = test[original_headers[1]].tolist()
test_y = test[original_headers[2]].tolist()

# #############################################################################
# Extracting features from text files
# #############################################################################

# Tokenizing text
# https://stackabuse.com/text-classification-with-python-and-scikit-learn/
count_vect = CountVectorizer(max_features=1500, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
X_train_counts = count_vect.fit_transform(train_x)

# #############################################################################
# Apply TFIF
# #############################################################################
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# #############################################################################
# Create a simple naive bayes
# #############################################################################
clf = MultinomialNB().fit(X_train_tfidf, train_y)

# #############################################################################
# Test simple naive bayes
# #############################################################################
X_test_counts = count_vect.transform(test_x)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    
clf_predicted = clf.predict(X_test_tfidf)

print("Accuracy: {0:.2f} %".format(100 * accuracy_score(test_y, clf_predicted)))

# #############################################################################
# Save model to a pickle file
# #############################################################################
with open(output_file, 'wb') as file:
    pickle.dump(clf, file)
print("Model file: "+output_file)