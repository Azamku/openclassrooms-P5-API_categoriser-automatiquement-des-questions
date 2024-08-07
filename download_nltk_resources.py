import nltk

# Spécifier le répertoire de téléchargement
nltk.data.path.append('./nltk_data')

# Télécharger les stopwords et le tokenizer de NLTK
nltk.download('stopwords', download_dir='./nltk_data')
nltk.download('punkt', download_dir='./nltk_data')
nltk.download('wordnet', download_dir='./nltk_data')
