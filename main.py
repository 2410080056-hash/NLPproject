from preprocessing import preprocess_text
from query_expansion import expand_query
from document_loader import load_documents
from search import build_search_index, search


def display_results(title, results):
    print(f"\n{title}")
    print("-" * len(title))

    for rank, (filename, score) in enumerate(results, start=1):
        print(f"{rank}. {filename} -> {score:.4f}")


def main():
    documents, filenames = load_documents("data")

    if not documents:
        print("No .txt documents found in the data folder.")
        return

    # Fit the vectorizer once and reuse it for both search models.
    vectorizer, document_vectors = build_search_index(documents)
    vocabulary = set(vectorizer.get_feature_names_out())

    query = input("Enter your search query: ")
    processed_query = preprocess_text(query)

    # Model 1: Baseline TF-IDF search (no WordNet expansion).
    baseline_results = search(
        processed_query, vectorizer, document_vectors, filenames
    )

    # Model 2: Proposed WordNet-expanded search.
    expanded_query = expand_query(processed_query, vocabulary)
    proposed_results = search(
        expanded_query, vectorizer, document_vectors, filenames
    )

    print("\nOriginal query:", query)
    print("Processed query:", processed_query)
    print("Expanded query:", expanded_query)

    display_results("BASELINE SEARCH (TF-IDF)", baseline_results)
    display_results("PROPOSED SEARCH (WordNet + TF-IDF)", proposed_results)


if __name__ == "__main__":
    main()
