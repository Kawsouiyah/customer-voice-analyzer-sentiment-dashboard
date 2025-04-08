import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Customer Voice Analyzer", layout="wide")

st.title("Customer Voice Analyzer - Sentiment Dashboard")

uploaded_file = st.file_uploader("📂 Importez un fichier CSV contenant les retours clients", type=["csv"])

analyzer = SentimentIntensityAnalyzer()

def correct_spelling(text):
    return str(TextBlob(text).correct())

def get_sentiment_vader(text):
    corrected_text = correct_spelling(text)  # Correction de l'orthographe
    sentiment_score = analyzer.polarity_scores(corrected_text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positif'
    elif sentiment_score['compound'] <= -0.05:
        return 'Négatif'
    else:
        return 'Neutre'

if uploaded_file:
    try:
        # Essayer avec un encodage ISO-8859-1 et ignorer les lignes mal formatées
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')
        
        # Vérifier si la colonne 'feedback' existe
        if 'feedback' in df.columns:
            # Appliquer l'analyse de sentiment avec VADER
            df['Sentiment'] = df['feedback'].apply(get_sentiment_vader)
            st.subheader("Aperçu des données")
            st.dataframe(df[['feedback', 'Sentiment']])

            sentiment_counts = df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Nombre']

            fig = px.pie(sentiment_counts, names='Sentiment', values='Nombre', title='Répartition des sentiments')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tester un nouveau message client")
            user_input = st.text_area("Entrez un message client ici")
            if user_input:
                label = get_sentiment_vader(user_input)
                st.markdown(f"**Sentiment détecté : {label}**")

            st.download_button("📥 Télécharger les résultats", data=df.to_csv(index=False).encode('utf-8'), file_name="resultats_sentiment.csv")
        else:
            st.error("La colonne 'feedback' est manquante dans le fichier CSV.")
    
    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez importer un fichier CSV contenant une colonne 'feedback'.")
