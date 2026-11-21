import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """Normalize text for both documents and user queries."""
    text = text.lower()
    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word.isalnum() and word not in stop_words
    ]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)
