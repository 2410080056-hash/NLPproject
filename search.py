from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def search(query, documents, filenames):

    # Create TF-IDF vectors for documents
    vectorizer = TfidfVectorizer()

    document_vectors = vectorizer.fit_transform(documents)

    # Create TF-IDF vector for the query
    query_vector = vectorizer.transform([query])

    # Calculate similarity
    scores = cosine_similarity(
        query_vector,
        document_vectors
    )[0]

    # Combine filenames and scores
    ranked_results = sorted(
        zip(filenames, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results