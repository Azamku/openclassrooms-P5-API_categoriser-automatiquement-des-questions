## script qui permet l'installation du package en_core_web_sm utilisé dans le pretraitement de texte
import spacy

def download_model():
    spacy.cli.download("en_core_web_sm")

if __name__ == "__main__":
    download_model()
