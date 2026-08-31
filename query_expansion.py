from synonyms import get_synonyms


def expand_query(query):

    words = query.split()

    expanded_words = set(words)

    for word in words:

        synonyms = get_synonyms(word)

        expanded_words.update(synonyms)

    return " ".join(expanded_words)