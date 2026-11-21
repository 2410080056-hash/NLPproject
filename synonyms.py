from nltk.corpus import wordnet


def get_synonyms(word):
    """Return normalized single-word WordNet synonyms for a term."""
    synonyms = set()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            candidate = lemma.name().replace("_", " ").lower().strip()

            # This project uses a unigram TF-IDF vocabulary.
            # Keep single-word candidates only.
            if candidate.isalpha() and candidate != word.lower():
                synonyms.add(candidate)

    return synonyms
