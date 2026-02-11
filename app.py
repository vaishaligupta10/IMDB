import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("IMDB Sentiment Analysis")
file = st.file_uploader("Upload IMDB Dataset CSV", type="csv")
if file:
    df = pd.read_csv(file)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    X = df['review']
    y = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    nb_model = MultinomialNB()
    nb_model.fit(X_train_tfidf, y_train)
    y_pred = nb_model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap( cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive'],
        ax=ax )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
    st.subheader("Try Your Own Review")
    user_review = st.text_area("Enter a movie review:")
    if st.button("Predict Sentiment"):
        if user_review.strip():
            review_tfidf = tfidf.transform([user_review])
            prediction = nb_model.predict(review_tfidf)
            sentiment = "Positive" if prediction[0] == 1 else "Negative"
            st.write(f"Prediction: {sentiment}")
fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='mako', ax=ax)
    ax.set_title("Sentiment Analysis Confusion Matrix")
    st.pyplot(fig)
