import nltk
import os

# Spécifier le répertoire de téléchargement
nltk_data_dir = './nltk_data'
nltk.data.path.append(nltk_data_dir)

# Fonction pour vérifier l'existence des fichiers NLTK téléchargés
def check_nltk_resource(resource_name, resource_path):
    if os.path.exists(resource_path):
        print(f"{resource_name} est déjà téléchargé.")
    else:
        print(f"Erreur: {resource_name} n'a pas été téléchargé.")

# Télécharger une ressource NLTK si elle n'est pas déjà présente
def download_if_not_present(resource_name):
    resource_path = os.path.join(nltk_data_dir, resource_name)
    if not os.path.exists(resource_path):
        print(f"Téléchargement de {resource_name}...")
        nltk.download(resource_name, download_dir=nltk_data_dir)
        check_nltk_resource(resource_name, resource_path)
    else:
        print(f"{resource_name} est déjà présent.")

# Télécharger les stopwords, punkt, et wordnet
download_if_not_present('corpora/stopwords.zip')
download_if_not_present('tokenizers/punkt.zip')
download_if_not_present('corpora/wordnet.zip')

print("Tous les téléchargements sont terminés.")
