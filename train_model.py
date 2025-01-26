import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from utils import TextCleaner
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Connect to database
conn = sqlite3.connect('imdb_reviews.db')

# Load data
train_df = pd.read_sql_query("SELECT review_text, sentiment FROM imdb_reviews WHERE split='train'", conn)
test_df = pd.read_sql_query("SELECT review_text, sentiment FROM imdb_reviews WHERE split='test'", conn)

# Convert sentiment to binary
train_df['sentiment'] = train_df['sentiment'].map({'positive': 1, 'negative': 0})
test_df['sentiment'] = test_df['sentiment'].map({'positive': 1, 'negative': 0})

# Define pipeline
pipeline = Pipeline([
    ('cleaner', TextCleaner()),
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train model
pipeline.fit(train_df['review_text'], train_df['sentiment'])

# Evaluate on test set
y_pred = pipeline.predict(test_df['review_text'])
print(classification_report(test_df['sentiment'], y_pred))

# Save the model
joblib.dump(pipeline, 'sentiment_pipeline.pkl')

conn.close()