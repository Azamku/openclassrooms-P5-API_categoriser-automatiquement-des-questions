import streamlit as st
import spacy

st.title("Test Spacy Import")

# Charger le modèle spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Hello, world!")

# Afficher le texte analysé
st.write(doc.text)
