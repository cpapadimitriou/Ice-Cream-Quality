import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    stop = set(stopwords.words('english'))
    text = text.rstrip()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    punc_free = ''.join(ch for ch in text if ch not in set(string.punctuation))
    text = ' '.join([i.lower() for i in punc_free.split()])
    clean_text = ' '.join(token for token in text.split() if len(token)>2)
    clean_text = ' '.join([i for i in clean_text.split() if (i not in stop) and (not i.isdigit())])
    return clean_text


def lemmatize_text(text):
	lemma = WordNetLemmatizer()
	lemmatized_text = " ".join(lemma.lemmatize(word) for word in text.split())
	return lemmatized_text


def tfidf(data):
	tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', analyzer='word')
	scores = tfidf_vectorizer.fit_transform(data)
	features = tfidf_vectorizer.get_feature_names()
	return tfidf_vectorizer, scores, features