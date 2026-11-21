import os
from preprocessing import preprocess_text


def load_documents(folder_path):
    """Load .txt files in a stable order and preprocess their contents."""
    documents = []
    filenames = []

    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".txt"):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append(preprocess_text(text))
            filenames.append(filename)

    return documents, filenames
