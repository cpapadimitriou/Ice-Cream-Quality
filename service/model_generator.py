# Import libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import ast
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, roc_auc_score
from src.util import clean_text, lemmatize_text, tfidf
from sklearn.externals import joblib


# nltk.download('wordnet')
# nltk.download('stopwords')

# Get the dataset
df = pd.read_excel('ice_cream_data.xlsx')
df = df.rename(columns={'ingredients': 'recipe_ingredients'})

# Ingredients

# count vectorizer
vectorizer = CountVectorizer()
X_count = vectorizer.fit_transform(df['recipe_ingredients'])
countvector = X_count.toarray().astype(int)
df['ingedients_vector'] = countvector.tolist()

# # tfidf 
# tfidf_transformer = TfidfTransformer()
# X_tfidf = tfidf_transformer.fit_transform(X_count)
# df['ingedients_tfidf_vector'] = X_tfidf.toarray().tolist()
# ingredients = vectorizer.get_feature_names()

# Cooking Steps
df['cooking_steps_all'] = df['cooking_steps'].apply(lambda x: " ".join(ast.literal_eval(x)))
df['cooking_steps_clean'] = [clean_text(i) for i in df['cooking_steps_all']] 
df['cooking_steps_lemma'] = [lemmatize_text(i) for i in df['cooking_steps_clean']] 
tfidf_vectorizer, x_tfidf, features = tfidf(df['cooking_steps_lemma'])
df_cooking_steps = pd.DataFrame(x_tfidf.toarray(), columns=features)
df = pd.concat([df, df_cooking_steps], axis=1)

# Target
df['is_good_recipe']  = df['is_good_recipe'].astype(int)


# Split the dataset into features and labels
y = np.array(df['is_good_recipe'])
x_ingredients = np.array(df['ingedients_vector'].tolist()) # ingedients_tfidf_vector
x_cooking = df[features].to_numpy()
x = np.concatenate([x_ingredients, x_cooking], axis=1)
predictors = ingredients + features


# Split the dataset into training (80%) and testing (20%) data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Build the classifier and make prediction
rf = RandomForestClassifier(n_estimators=1000, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Print metrics
cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred), 
        columns=['Predicted=0', 'Predicted=1'], 
        index=['Actual=0', 'Actual=1']
)
print("Confusion Matrix:\n")
print(cm)
print("\n")

cr = classification_report(y_test, y_pred)
print("Classification Report:\n",)
print (cr)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy:", accuracy)


# Save the model to disk
joblib.dump(rf, 'classifier.joblib')





