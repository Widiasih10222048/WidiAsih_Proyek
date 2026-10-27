import re
import joblib
import pandas as pd
import streamlit as st
import saka

st.set_page_config(page_title="Analisis Sentimen Komentar", page_icon="💬", layout="centered")

@st.cache_resource
def load_model():
    svm = joblib.load("svm_model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return svm, tfidf

svm, tfidf = load_model()

NEGASI = {"tidak", "bukan", "belum", "tak", "kurang", "nggak", "gak", "jangan"}
STOPWORDS = set(saka.get_stopwords("id")) - NEGASI

def cleaning(text):
    text = str(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess(text):
    text = cleaning(text).lower()
    text = saka.normalize(text)
    tokens = [t for t in saka.tokenize(text) if t not in STOPWORDS]
    return " ".join(tokens)

st.title("💬 Analisis Sentimen Komentar")
st.caption("Model: SVM + TF-IDF | Preprocessing: Saka-NLP")

tab1, tab2 = st.tabs(["Satu komentar", "Upload file"])

with tab1:
    komentar = st.text_area("Tulis komentar:", height=120,
                            placeholder="contoh: pelayanannya tidak bagus dan mengecewakan")
    if st.button("Prediksi", type="primary"):
        if komentar.strip():
            bersih = preprocess(komentar)
            label = svm.predict(tfidf.transform([bersih]))[0]
            warna = {"positif": "green", "negatif": "red"}.get(label, "gray")
            st.markdown(f"### Sentimen: :{warna}[{label.upper()}]")
            with st.expander("Lihat hasil preprocessing"):
                st.write(bersih if bersih else "(kosong setelah preprocessing)")
        else:
            st.warning("Komentar tidak boleh kosong.")

with tab2:
    file = st.file_uploader("Upload CSV atau Excel", type=["csv", "xlsx"])
    if file:
        df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)

        kandidat = ["content", "review", "comment", "komentar", "ulasan", "text", "teks"]
        kolom_lower = [c.lower() for c in df.columns]
        default = next((kolom_lower.index(k) for k in kandidat if k in kolom_lower), 0)

        kolom = st.selectbox("Pilih kolom komentar:", df.columns, index=default)
        st.dataframe(df[[kolom]].head())   

        if st.button("Proses semua", type="primary"):
            texts = df[kolom].astype(str).tolist()
            bersih = []
            bar = st.progress(0, text="Preprocessing...")
            for i, t in enumerate(texts):
                bersih.append(preprocess(t))
                if i % 50 == 0 or i == len(texts) - 1:
                    bar.progress((i + 1) / len(texts), text=f"Preprocessing {i + 1}/{len(texts)}")
            bar.empty()

            with st.spinner("Memprediksi..."):
                hasil = svm.predict(tfidf.transform(bersih))

            df["text_clean"] = bersih
            df["sentimen"] = hasil

            st.success("Selesai!")
            st.bar_chart(df["sentimen"].value_counts())
            st.write(df["sentimen"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")
            st.dataframe(df)

            st.download_button(
                "⬇️ Download hasil (CSV)",
                df.to_csv(index=False).encode("utf-8"),
                file_name="hasil_sentimen.csv",
                mime="text/csv",
            )