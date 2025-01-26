from datasets import load_dataset
import sqlite3

# Load the IMDB dataset
dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']

# Connect to SQLite database
conn = sqlite3.connect('imdb_reviews.db')
cursor = conn.cursor()

# Create table with split column
cursor.execute('''
    CREATE TABLE IF NOT EXISTS imdb_reviews (
        id INTEGER PRIMARY KEY,
        review_text TEXT,
        sentiment TEXT,
        split TEXT
    )
''')

# Prepare data for insertion
def prepare_data(data, split):
    return [(example['text'], 'positive' if example['label'] else 'negative', split) for example in data]

train_records = prepare_data(train_data, 'train')
test_records = prepare_data(test_data, 'test')

# Insert data
cursor.executemany('''
    INSERT INTO imdb_reviews (review_text, sentiment, split)
    VALUES (?, ?, ?)
''', train_records + test_records)

conn.commit()
conn.close()