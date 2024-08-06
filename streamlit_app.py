import streamlit as st
import spacy
import numpy
import pandas as pd

st.title("Test Spacy Import")

# Afficher les versions des bibliothèques
st.write(f"Version de Spacy: {spacy.__version__}")
st.write(f"Version de Numpy: {numpy.__version__}")
st.write(f"Version de Pandas: {pd.__version__}")


