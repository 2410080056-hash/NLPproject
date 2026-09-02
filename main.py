from preprocessing import preprocess_text
from query_expansion import expand_query
from document_loader import load_documents
from search import search


# --------------------------------
# 1. Load documents
# --------------------------------

documents, filenames = load_documents("data")


# --------------------------------
# 2. Get user query
# --------------------------------

query = input("Enter your search query: ")


# --------------------------------
# 3. Preprocess query
# --------------------------------

processed_query = preprocess_text(query)


# --------------------------------
# 4. Expand query using WordNet
# --------------------------------

expanded_query = expand_query(processed_query)


# --------------------------------
# 5. Search documents
# --------------------------------

results = search(
    expanded_query,
    documents,
    filenames
)


# --------------------------------
# 6. Display information
# --------------------------------

print("\nOriginal Query:")
print(query)

print("\nProcessed Query:")
print(processed_query)

print("\nExpanded Query:")
print(expanded_query)


# --------------------------------
# 7. Display ranked results
# --------------------------------

print("\nSearch Results:")
print("----------------------------")

for filename, score in results:

    print(f"{filename} -> {score:.4f}")


    #main