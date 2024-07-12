import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    text = text.lower().strip()
    return text

df = pd.read_csv('data/tweets.csv')
df['cleaned_text'] = df['text'].apply(clean_text)

# Adding a dummy sentiment column for demonstration purposes
df['sentiment'] = df['text'].apply(lambda x: 1 if 'good' in x else 0)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned_text'], df['sentiment'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Save the processed data
import pickle
pickle.dump((X_train_vec, X_test_vec, y_train, y_test, vectorizer), open('data/processed_data.pkl', 'wb'))