from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_search_index(documents):
    """Fit one TF-IDF model on the document collection."""
    vectorizer = TfidfVectorizer()
    document_vectors = vectorizer.fit_transform(documents)
    return vectorizer, document_vectors


def search(query, vectorizer, document_vectors,
           filenames, top_k=5):

    query_vector = vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        document_vectors
    ).flatten()

    ranked_indices = scores.argsort()[::-1]

    results = []

    for index in ranked_indices:

        if scores[index] <= 0:
            continue

        results.append(
            (filenames[index], float(scores[index]))
        )

        if len(results) == top_k:
            break

    return results
