import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords')

# stop_words_indonesian = set(stopwords.words("indonesian"))

from huggingface_hub import login
login("TOKEN_HUGGINGFACE_ANDA")  # Ganti dengan token kamu

# Load tokenizer and model IndoBERTweet
MODEL_NAME = "model/indobert_bilstm"
# MODEL_NAME = "Aardiiiiy/indobertweet-base-Indonesian-sentiment-analysis"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="./cache")
model = TFAutoModelForSequenceClassification.from_pretrained(MODEL_NAME, cache_dir="./cache")


# Label Encoder (sesuaikan dengan label model Anda)
labels = ['Negatif', 'Netral', 'Positif']
le = LabelEncoder()
le.fit(labels)

# Preprocessing Function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

MAX_LEN = 128
def preprocess_tf(texts):
    return tokenizer(
        texts,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

# Predict Function
def predict_sentiment(text):
    enc = preprocess_tf([text])
    pred = model(enc)
    probs = tf.nn.softmax(pred.logits, axis=1).numpy()[0]
    idx = np.argmax(probs)
    return labels[idx], float(probs[idx])


# Streamlit layout
st.set_page_config(page_title="Sentiment Analysis IndoBERTweet", layout="wide")
st.title("🇮🇩 Analisis Sentimen dengan IndoBERTweet")

# Upload CSV untuk EDA
st.header("📊 Exploratory Data Analysis (EDA) Dataset")
try:
    df = pd.read_csv("data.csv")

    if 'Sentiment' in df.columns and 'clean_tweet' in df.columns:
        df['cleaned'] = df['clean_tweet'].astype(str).apply(clean_text)

        st.subheader("Distribusi Label Sentimen")
        st.bar_chart(df['Sentiment'].value_counts())

        st.subheader("☁️ Word Cloud per Sentimen")
        for sentiment in df['Sentiment'].unique():
            st.markdown(f"**{sentiment}**")
            text = " ".join(df[df['Sentiment'] == sentiment]['cleaned'])
            wordcloud = WordCloud(
                stopwords=set(stopwords.words("indonesian")),
                background_color="white", width=600, height=200
            ).generate(text)
            fig, ax = plt.subplots()
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        st.subheader("🧮 Kata Paling Umum")
        all_words = " ".join(df['cleaned']).split()
        common_words = Counter(all_words).most_common(15)
        common_df = pd.DataFrame(common_words, columns=['Word', 'Frequency'])
        st.bar_chart(common_df.set_index('Word'))

        st.subheader("📏 Distribusi Panjang Tweet")
        df['tweet_length'] = df['clean_tweet'].astype(str).apply(len)
        fig, ax = plt.subplots()
        sns.histplot(df['tweet_length'], bins=30, kde=True, ax=ax)
        ax.set_title("Distribusi Panjang Tweet")
        st.pyplot(fig)

        st.subheader("🔍 Contoh Data")
        st.dataframe(df[['clean_tweet', 'Sentiment']].sample(10))

        st.subheader("📈 Statistik Deskriptif")
        st.write(df.describe(include='all'))

        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(stop_words=stopwords.words("indonesian"), max_features=20)
        X = vectorizer.fit_transform(df['cleaned'])
        word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        word_freq['Sentiment'] = df['Sentiment']

        st.subheader("🧯 Korelasi Kata Populer dan Sentimen")
        pivot = word_freq.groupby('Sentiment').mean().T
        fig, ax = plt.subplots(figsize=(10,5))
        sns.heatmap(pivot, annot=True, cmap='YlGnBu')
        st.pyplot(fig)

        vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=10, stop_words=stopwords.words("indonesian"))
        X = vectorizer.fit_transform(df['cleaned'])
        top_bigrams = pd.DataFrame({'Bigram': vectorizer.get_feature_names_out(), 'Frequency': X.toarray().sum(axis=0)})
        st.subheader("🧩 Top 10 Bigram")
        st.dataframe(top_bigrams.sort_values("Frequency", ascending=False))

    else:
        st.error("File 'data.csv' harus memiliki kolom 'clean_tweet' dan 'Sentiment'.")

except FileNotFoundError:
    st.error("File 'data.csv' tidak ditemukan. Pastikan file berada di direktori yang sama dengan app.py.")


# Prediksi Sentimen Manual
st.header("🔮 Prediksi Sentimen Manual")
input_text = st.text_area("Masukkan tweet atau kalimat:")

if st.button("Prediksi"):
    if input_text.strip():
        label, confidence = predict_sentiment(input_text)
        st.success(f"Sentimen: **{label}** ({confidence:.2f})")
    else:
        st.warning("Masukkan teks terlebih dahulu.")    