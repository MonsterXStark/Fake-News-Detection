import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import nltk
import gradio as gr

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

# Function to predict on new text
def predict_news(text):
    text_vec = vectorizer.transform([preprocess_text(text)])
    prediction = model.predict(text_vec)
    if prediction[0] == 0:
        return "Fake News"
    else:
        return "Real News"

# Define the Gradio interface
iface = gr.Interface(
    fn=predict_news,
    inputs=gr.Textbox(),
    outputs=gr.Textbox(),
    live=True
    )

# Launch the Gradio interface
iface.launch()
