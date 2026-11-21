"""
Evaluate baseline TF-IDF and WordNet-expanded search.

Ground-truth labels below are an initial manual relevance assessment
based on the contents of this project's ten supplied documents.
The project team should review and approve these labels before
using the resulting metrics in the final report.
"""

import pandas as pd

from document_loader import load_documents
from preprocessing import preprocess_text
from query_expansion import expand_query
from search import build_search_index, search


# Each query maps to the documents judged relevant for that query.
# These are document-level relevance judgments, not model predictions.
GROUND_TRUTH = {
    "car": ["doc1.txt", "doc2.txt"],
    "car maintenance": ["doc1.txt", "doc2.txt"],
    "healthcare": ["doc7.txt"],
    "machine learning": ["doc3.txt", "doc4.txt", "doc9.txt"],
    "educational technology": ["doc8.txt"],
    "vehicle repair": ["doc1.txt", "doc2.txt"],
    "solar energy": ["doc10.txt"],
    "renewable energy": ["doc10.txt"],
    "digital learning": ["doc8.txt"],
    "technology security": ["doc6.txt"],
    "artificial intelligence": ["doc3.txt", "doc7.txt", "doc8.txt"],
    "computers": ["doc3.txt", "doc5.txt", "doc6.txt"],
    "power": ["doc10.txt"],
    "weather forecasting": ["doc9.txt"]
}


def calculate_metrics(retrieved, relevant):
    """Calculate precision, recall and F1 for one query."""
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    true_positive = len(retrieved_set & relevant_set)
    false_positive = len(retrieved_set - relevant_set)
    false_negative = len(relevant_set - retrieved_set)

    precision = (
        true_positive / len(retrieved_set)
        if retrieved_set else 0.0
    )
    recall = (
        true_positive / len(relevant_set)
        if relevant_set else 0.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def evaluate_model(model_name, use_wordnet, vectorizer,
                   document_vectors, filenames, vocabulary, top_k=5):
    rows = []

    for query, relevant_docs in GROUND_TRUTH.items():
        processed_query = preprocess_text(query)

        if use_wordnet:
            search_query = expand_query(processed_query, vocabulary)
        else:
            search_query = processed_query

        results = search(
            search_query, vectorizer, document_vectors, filenames, top_k
        )
        retrieved_docs = [filename for filename, score in results]

        precision, recall, f1 = calculate_metrics(
            retrieved_docs, relevant_docs
        )

        rows.append({
            "Model": model_name,
            "Query": query,
            "Retrieved": ", ".join(retrieved_docs),
            "Relevant": ", ".join(relevant_docs),
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
        })

    return pd.DataFrame(rows)


def main():
    documents, filenames = load_documents("data")

    if not documents:
        print("No .txt documents found in the data folder.")
        return

    vectorizer, document_vectors = build_search_index(documents)
    vocabulary = set(vectorizer.get_feature_names_out())

    baseline = evaluate_model(
        "Baseline TF-IDF", False, vectorizer,
        document_vectors, filenames, vocabulary
    )
    proposed = evaluate_model(
        "WordNet + TF-IDF", True, vectorizer,
        document_vectors, filenames, vocabulary
    )

    details = pd.concat([baseline, proposed], ignore_index=True)

    print("\nPER-QUERY EVALUATION")
    print(details.to_string(index=False))

    summary = details.groupby("Model")[
        ["Precision", "Recall", "F1-score"]
    ].mean()

    print("\nMACRO-AVERAGE COMPARISON")
    print(summary.round(4).to_string())

    details.to_csv("evaluation_details.csv", index=False)
    summary.to_csv("evaluation_summary.csv")

    print("\nSaved evaluation_details.csv and evaluation_summary.csv")


if __name__ == "__main__":
    main()
