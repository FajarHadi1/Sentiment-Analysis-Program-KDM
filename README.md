# 🇮🇩 Analisis Sentimen Publik terhadap Barak Militer KDM

Aplikasi ini dibuat untuk menganalisis sentimen masyarakat Indonesia terhadap kebijakan **Kang Dedi Mulyadi** yang memasukkan anak-anak ke dalam **barak militer**. Proyek ini menggunakan **Natural Language Processing (NLP)** dengan model **IndoBERTweet-BiLSTM** dan dibangun menggunakan **Streamlit** sebagai antarmuka pengguna interaktif.

---

## 🎯 Fitur Utama

- 🔍 **Exploratory Data Analysis (EDA)**  
  Visualisasi distribusi sentimen, kata populer, korelasi kata terhadap emosi, dan analisis bigram.

- 📝 **Prediksi Sentimen**  
  Input teks manual untuk analisis real-time  

- 🤖 **Model Kustom**  
  Model klasifikasi sentimen berbasis **IndoBERTweet + BiLSTM** yang sudah dilatih dan disimpan secara lokal.

---

## 🧠 Teknologi yang Digunakan

- `Transformers` – IndoBERTweet dari HuggingFace
- `TensorFlow` – untuk BiLSTM dan klasifikasi
- `Scikit-learn`, `Seaborn`, `Matplotlib`, `NLTK`
- `Streamlit` – untuk tampilan web interaktif

---

## 🗂️ Struktur Proyek

```
sentimen-app/
├── app.py                  # Aplikasi Streamlit
├── train_model.py          # Script pelatihan model
├── model/
│   └── indobert_bilstm/    # Model hasil training (config, tokenizer, tf_model.h5)
├── data.csv                # Dataset tweet (bersih & berlabel)
├── requirements.txt        # Daftar dependency
├── .gitignore
└── README.md
```

---

## ⬇️ Unduh Model Terlatih

Model `IndoBERTweet-BiLSTM` yang telah dilatih **tidak diunggah ke GitHub** karena ukurannya melebihi batas maksimal. Kamu bisa mengunduhnya melalui Google Drive.

### 🔗 Link Unduh Model

📥 [Download model/indobert_bilstm.zip](https://drive.google.com/drive/folders/1MO8Zr916159tQS-KtupWLiz4bRsr8PVY?usp=sharing)

### 📁 Cara Menggunakan

1. Unduh file ZIP dari link di atas.
2. Ekstrak file tersebut, maka akan muncul folder `indobert_bilstm`.
3. Letakkan folder tersebut ke dalam folder `model/` di struktur proyek:

```
sentimen-app/
├── model/
│   └── indobert_bilstm/
│       ├── config.json
│       ├── tokenizer_config.json
│       ├── vocab.txt
│       ├── tf_model.h5
```

4. Setelah itu, kamu bisa langsung menjalankan aplikasi dengan Streamlit.

---

## 🚀 Cara Menjalankan Aplikasi

1. **Clone repositori dan buat virtual environment**:

```bash
git clone https://github.com/username/sentimen-app.git
cd sentimen-app
python -m venv venv
venv\Scripts\activate        # Jika kamu menggunakan Windows
pip install -r requirements.txt
```

2. **Jalankan Streamlit**:

```bash
streamlit run app.py
```

3. **Gunakan fitur EDA atau lakukan prediksi sentimen dari teks/manual/CSV**

---

## 📊 Output Visualisasi

- Bar chart distribusi sentimen
- Wordcloud untuk tiap label
- Heatmap korelasi kata terhadap sentimen
- Top 10 Bigram berdasarkan frekuensi

---

## 🤖 Tentang Model

Model ini menggunakan pendekatan:
- Tokenisasi dengan **IndoBERTweet tokenizer**
- Embedding BERT + lapisan **BiLSTM**
- Klasifikasi ke dalam 3 label: Positif, Netral, dan Negatif

Model dilatih dengan dataset bersih dan disimpan di `model/indobert_bilstm/`.

---

## 📄 Dataset

Dataset berupa kumpulan tweet berbahasa Indonesia seputar isu **barak militer dan KDM**. Data telah melalui proses preprocessing dan anotasi sentimen secara manual.

Kolom penting:
- `clean_tweet`: teks tweet setelah preprocessing
- `Sentiment`: label sentimen (`positif`, `netral`, `negatif`)

---

## 🖥️ Preview

<img width="1920" height="1020" alt="Screenshot 2025-07-20 215031" src="https://github.com/user-attachments/assets/64e95919-9b15-4fb7-963c-218d63f34cca" />
<img width="1920" height="1020" alt="Screenshot 2025-<img width="1920" height="1020" alt="Screenshot 2025-07-20 215041" src="https://github.com/user-attachments/assets/01d45d57-a109-4441-be37-076f26319ff3" />
<img width="1920" height="1020" alt="Screenshot 2025-07-21 062159" src="https://github.com/user-attachments/assets/f894a4b6-72b8-4243-a3de-1dd75eb72bcc" />

---

## ✍️ Author

**Farras Fajar Hadi, Rosalin Keyzia Constantia, Shandika Eka Revananda**  
Mahasiswa Politeknik Caltex Riau  
AI & NLP Enthusiast – 2025

---

## 📄 Lisensi

MIT License – Bebas digunakan, disebarluaskan, dan dimodifikasi untuk keperluan edukasi maupun penelitian.
