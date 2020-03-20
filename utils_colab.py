import nltk
import pickle
import re
import numpy as np

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': '/content/data/intent_recognizer.pkl',
    'TAG_CLASSIFIER': '/content/data/tag_classifier.pkl',
    'TFIDF_VECTORIZER': '/content/data/tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': '/content/data/thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': '/content/data/word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""

    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()



def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.
    Args:
      embeddings_path - path to the embeddings file.
    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    import numpy as np
    embeddings = {}
    with open(embeddings_path) as f: 
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vec
    embeddings_dim=len(vec)
    return embeddings,embeddings_dim
   
def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings.
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
    word_list=question.split()
    
    kword_list=[wv for wv in word_list if wv in embeddings]

    output_vector=np.zeros(dim)
    if kword_list:
               output_vector= np.mean(np.vstack([embeddings[wv] for wv in kword_list]),axis=0)
    return output_vector

def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
