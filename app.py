import streamlit as st
import pandas as pd
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px

st.set_page_config(page_title="Analyse Sentimentale - DXC", layout="wide")
st.title("📊 Analyse des Retours Clients - Sentiment Dashboard")

uploaded_file = st.file_uploader("📂 Importez un fichier CSV contenant les retours clients", type=["csv"])

analyzer = SentimentIntensityAnalyzer()

def correct_spelling(text):
    return str(TextBlob(text).correct())

def get_sentiment_vader(text):
    corrected_text = correct_spelling(text)
    sentiment_score = analyzer.polarity_scores(corrected_text)
    if sentiment_score['compound'] >= 0.05:
        return 'Positif'
    elif sentiment_score['compound'] <= -0.05:
        return 'Neutre'
    else:
        return 'Négatif'

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, encoding='ISO-8859-1', on_bad_lines='skip')

        if 'feedback' in df.columns:
            df['Sentiment'] = df['feedback'].apply(get_sentiment_vader)

            st.subheader("✏️ Modifier les sentiments directement dans le tableau")

            edited_df = st.data_editor(
                df[['feedback', 'Sentiment']],
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "Sentiment": st.column_config.SelectboxColumn(
                        "Sentiment",
                        help="Modifiez le sentiment si besoin",
                        options=["Positif", "Neutre", "Négatif"]
                    )
                }
            )

            sentiment_counts = edited_df['Sentiment'].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Nombre']

            fig = px.pie(sentiment_counts, names='Sentiment', values='Nombre', title='Répartition des sentiments')
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Tester un nouveau message client")
            user_input = st.text_area("Entrez un message client ici")
            if user_input:
                label = get_sentiment_vader(user_input)
                st.markdown(f"**Sentiment détecté : `{label}`**")

            st.download_button("📥 Télécharger les résultats modifiés", data=edited_df.to_csv(index=False).encode('utf-8'), file_name="resultats_sentiment.csv")

        else:
            st.error("La colonne 'feedback' est manquante dans le fichier CSV.")

    except Exception as e:
        st.error(f"Erreur lors du traitement du fichier : {e}")
else:
    st.info("Veuillez importer un fichier CSV contenant une colonne 'feedback'.")
