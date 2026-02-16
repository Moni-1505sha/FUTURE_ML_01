print("Program Started ")

import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df = pd.read_csv("customer_support_tickets.csv")
print("Dataset Loaded Successfully")

print("\nColumns:")
print(df.columns)

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

df['clean_text'] = df['Ticket Description'].apply(clean_text)

print("\nText Cleaning Completed")

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['clean_text'])

print("TF-IDF Completed")

y_category = df['Ticket Type']

X_train, X_test, y_train, y_test = train_test_split(
  X, y_category, test_size=0.2, random_state=42
)

model_category = LogisticRegression(max_iter=1000)
model_category.fit(X_train, y_train)

pred_category = model_category.predict(X_test)

print("\n===== Ticket Type Classification =====")
print(classification_report(y_test, pred_category))
print("Accuracy:", accuracy_score(y_test, pred_category))

y_priority = df['Ticket Priority']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X, y_priority, test_size=0.2, random_state=42
)

model_priority = LogisticRegression(max_iter=1000)
model_priority.fit(X_train_p, y_train_p)

pred_priority = model_priority.predict(X_test_p)

print("\n===== Ticket Priority Classification =====")
print(classification_report(y_test_p, pred_priority))
print("Accuracy:", accuracy_score(y_test_p, pred_priority))

print("\nModel Training Completed Successfully")
