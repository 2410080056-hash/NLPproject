import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Load English stopwords
stop_words = set(stopwords.words("english"))

# Create lemmatizer
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Tokenize text
    tokens = word_tokenize(text)

    # 3. Remove punctuation and non-alphanumeric tokens
    tokens = [
        word for word in tokens
        if word.isalnum()
    ]

    # 4. Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # 5. Lemmatize words
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)
