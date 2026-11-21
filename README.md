# Smarter Search Engine – WordNet Query Expansion

## Project contents
- `data/`: ten text documents used as the search collection
- `preprocessing.py`: text normalization
- `synonyms.py`: WordNet synonym extraction
- `query_expansion.py`: controlled query expansion
- `search.py`: shared TF-IDF index and cosine-similarity retrieval
- `main.py`: interactive baseline and proposed search demo
- `evaluation.py`: Precision, Recall and F1-score comparison

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download required NLTK resources once:

```python
import nltk
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource)
```

If your NLTK version does not require `punkt_tab`, downloading it is harmless.

## Run the interactive search

From the project root:

```bash
python main.py
```

The program displays both:
1. Baseline TF-IDF search (without WordNet)
2. Proposed WordNet-expanded TF-IDF search

## Run evaluation

```bash
python evaluation.py
```

The evaluation uses the same document collection, TF-IDF vectorizer, test queries and top-5 cutoff for both models. It produces:
- `evaluation_details.csv`
- `evaluation_summary.csv`

## Important evaluation note

`GROUND_TRUTH` in `evaluation.py` contains initial manually assigned relevance judgments based on the supplied documents. The project team must review these judgments and edit them if necessary before reporting results. Metrics are meaningful only when relevance labels are carefully judged independently of the search results.

The reported averages are macro-averages across the test queries. Precision uses the number of returned top-k documents as its denominator.
