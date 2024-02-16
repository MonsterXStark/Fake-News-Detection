import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import nltk

# Download NLTK stopwords data
nltk.download('stopwords')

# Define dataset path and label column
data_path = r"C:\Users\karne\Downloads\train.csv"
label_column = "label"

# Read data from CSV
df = pd.read_csv(data_path)

# Preprocess text data
def preprocess_text(text):
    # Check if the input is a string
    if isinstance(text, str):
        # Lowercase, remove punctuation, remove stopwords
        text = text.lower()
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        from nltk.corpus import stopwords
        stop_words = stopwords.words("english")
        text = " ".join([word for word in text.split() if word not in stop_words])
        return text
    else:
        # Return an empty string for NaN or non-string values
        return ""

df["text"] = df["text"].apply(preprocess_text)

# Drop rows with NaN values in the "text" column
df = df.dropna(subset=['text'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df[label_column], test_size=0.2, random_state=42)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train the Passive Aggressive Classifier
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate model performance
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Save the trained model
pickle.dump(model, open("fake_news_model.pkl", "wb"))

# Function to predict on new text
def predict_news(text):
    text_vec = vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(text_vec)
    if prediction[0] == 0:
        print("Prediction: Fake News")
    else:
        print("Prediction: Real News")

# Example usage
new_text = "This is a very important and groundbreaking news article that everyone should read!"
predict_news(new_text)
