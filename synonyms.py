from nltk.corpus import wordnet


def get_synonyms(word):
    synonyms = set()

    for synset in wordnet.synsets(word):
        for lemma in synset.lemmas():
            synonym = lemma.name().replace("_", " ")
            synonyms.add(synonym)

    return synonyms

