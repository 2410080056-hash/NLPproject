from synonyms import get_synonyms
from preprocessing import preprocess_text


def expand_query(query, vocabulary, max_synonyms_per_word=3):
    """
    Expand a preprocessed query with WordNet synonyms that occur
    in the document vocabulary. Original query terms are retained.
    """
    words = preprocess_text(query).split()
    expanded_words = list(words)

    for word in words:
        candidates = get_synonyms(word)

        # Only add terms the TF-IDF model knows about.
        useful_candidates = sorted(
            candidate for candidate in candidates
            if candidate in vocabulary
        )

        expanded_words.extend(useful_candidates[:max_synonyms_per_word])

    # Preserve order and remove duplicates.
    return " ".join(dict.fromkeys(expanded_words))
