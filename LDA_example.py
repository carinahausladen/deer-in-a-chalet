import gensim
import gensim.corpora as corpora
from gensim.models import LdaModel
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk

# Download required resources (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

documents = [
    "Die Regierung plant eine neue Steuerreform und diskutiert sie im Bundestag.",
    "Der Kanzler spricht über die wirtschaftliche Lage und neue Gesetze.",
    "Im Parlament gibt es eine Debatte über soziale Gerechtigkeit und Mindestlohn.",
    "Die globale Erwärmung bedroht unser Klima. Wissenschaftler fordern Maßnahmen.",
    "Klimaschutz ist wichtig. Neue CO2-Gesetze sollen die Umwelt verbessern."
]

# Preprocess function: Tokenization, stopword removal, and punctuation removal
def preprocess(text):
    stop_words = set(stopwords.words('german'))  # German stopwords
    tokens = word_tokenize(text.lower())  # Tokenize & lowercase
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]  # Remove stopwords/punctuation
    return tokens

# Apply preprocessing to all documents
texts = [preprocess(doc) for doc in documents]

# Create dictionary and corpus
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# Train LDA model
lda_model = LdaModel(corpus,
                     num_topics = 2,  # Number of topics to extract,
                     id2word=dictionary,
                     passes=10)

# Print topics
for idx, topic in lda_model.print_topics():
    print(f"Topic {idx}: {topic}")
