import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the dataset
data = pd.read_csv('/Users/berkebarantozkoparan/Desktop/spam detection/spam.csv', encoding='latin-1')

# Preprocess the text data
data['processed_text'] = data['Message'].apply(preprocess_text)

# Encode labels: spam=1, ham=0
data['label'] = data['Category'].map({'spam': 1, 'ham': 0})

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['processed_text'], data['label'], test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluate
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Save model and vectorizer
joblib.dump({'model': model, 'vectorizer': vectorizer}, 'model.pkl')
print("Model kaydedildi: model.pkl")
