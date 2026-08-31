import os

from preprocessing import preprocess_text


def load_documents(folder_path):

    documents = []
    filenames = []

    for filename in os.listdir(folder_path):

        if filename.endswith(".txt"):

            file_path = os.path.join(folder_path, filename)

            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

            # Preprocess document
            processed_text = preprocess_text(text)

            documents.append(processed_text)
            filenames.append(filename)

    return documents, filenames

