import logging
import os
import sys

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess


def read_data(filename):
    """
    Args:
        filename: filename of the data that will be turned into documents.
                  the data is expected to have each sentence of the text
                  separated by newlines, which paragraphs separated by two
                  newlines.

    Returns:
        A list of documents where each element is a list of words. Each sentence
        in the original data is turned into a document, as well as each paragraph.

    Raises:
        FileNotFoundException
    """

    data_file = open(filename, "r")

    # can't return generator because gensim requires this to be a list for
    # parallelisation speedups in training
    documents = []
    current_paragraph = []
    for line in data_file.readlines():
        # new paragraph
        if line == "\n":
            documents.append(current_paragraph)
            current_paragraph = []
        else:
            # strip trailing newline character and preprocess
            tokens = simple_preprocess(line.strip())
            documents.append(tokens)
            current_paragraph += tokens

    return documents


def create_model(documents, vector_size=100, min_count=2, workers=os.cpu_count()):
    """
    Args:
        documents: list of documents where each document is a list of words
        vector_size: dimensionality of vectors to represent each word. 100-150
                     should work well for wikipedia sized datasets
        min_count: minimum number of times a word must appear in order for
                   the model to not ignore it
        workers: number of threads to use

    Returns:
        gensim word2vec model trained on the documents
    """

    model = Word2Vec(sentences=documents,
                     vector_size=vector_size,
                     min_count=min_count,
                     workers=workers)

    return model


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    if len(sys.argv) < 3:
        print(
            "Usage: python train_model.py [document_filename] [model_filename]")
        sys.exit()

    filename = sys.argv[1]
    documents = read_data(filename)

    model = create_model(documents)

    model.save(sys.argv[2])
