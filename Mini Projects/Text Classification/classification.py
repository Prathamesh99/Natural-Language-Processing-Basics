!pip3 install nltk
import nltk
nltk.download()
import numpy as np
import re
import pickle
from sklearn.datasets import load_files
from nltk.corpus import stopwords

reviews = load_files("txt_sentoken/")
X, y = reviews.data, reviews.target

with open("X.pickle",'wb') as f:
	pickle.dump(X,f)
	
with open("y.pickle","wb") as f:
	pickle.dump(y,f)
	
corpus = []
for i in range(len(X)):
	review = re.sub(r'\W',' ',str(X[i]))
	review = review.lower()
	review = re.sub(r'\s+[a-z]\s+',' ',review)
	review = re.sub(r'^[a-z]\s+',' ',review)
	review = re.sub(r'\s+',' ',review)
	corpus.append(review)
	
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
vectorizer = CountVectorizer(max_features=2000, min_df=3, max_df=0.6, stop_words=stopwords.words('english'))
transformer = TfidfTransformer()
X = vectorizer.fit_transform(corpus).toarray()
X = transformer.fit_transform(X).toarray()
