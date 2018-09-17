import logging
from typing import List
from gensim.models import KeyedVectors
#from gensim import KeyedVectors
#from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
#
#
# class EmbeddingHandler:
#     """
#     Parameters
#     ----------
#     embedding_file : `str`
#         Location of a text file containing embeddings in word2vec format.
#     """
#     def __init__(self, embedding_file: str) -> None:
#         # str -> int
#         self.word_indices = {}
#         # int -> str
#         self.index_to_word = {}
#         # list of np arrays
#         embedding_weights = []
#
#         with open(embedding_file) as input_file:
#             idx = 0
#             expected_embedding_size = None
#             for line in input_file:
#                 fields = line.strip().split()
#                 if len(fields) == 2:
#                     # This must be the first line with metadata.
#                     expected_embedding_size = int(fields[1])
#                     continue
#                 if expected_embedding_size is not None and \
#                    len(fields[1:]) != expected_embedding_size:
#                     continue
#                 word = fields[0].lower()
#                 vector = np.asarray([float(x) for x in fields[1:]])
#                 embedding_weights.append(vector)
#                 self.word_indices[word] = idx
#                 self.index_to_word[idx] = word
#                 idx += 1
#                 print(idx)
#         self.embedding_weights = np.asarray(embedding_weights)
#         logging.info(len(self.embedding_weights))
#         logging.info(len(self.word_indices))
#         logging.info("Found embeddings for {} words".format(len(self.embedding_weights)))
#
#         assert len(self.embedding_weights) == len(self.word_indices)
#
#     def get_word_vector(self, word: str) -> np.ndarray:
#         word = word.lower()
#         if word not in self.word_indices:
#             return None
#         return self.embedding_weights[self.word_indices[word]]
#
#     def sort_options_by_similarity(self,
#                                    clue: str,
#                                    options: List[str],
#                                    max_num_outputs: int = -1) -> List[str]:
#         """
#         Takes a clue and returns `max_num_outputs` number of `options` sorted by similarity with
#         `clue`. If `max_num_outputs` is -1, returns all the options sorted by similarity.
#         """
#         word = clue.lower()
#         if word not in self.word_indices:
#             return []
#         word_vector = self.get_word_vector(word)
#         option_vectors = []
#         for option in options:
#             if option in self.word_indices:
#                 option_vectors.append(self.get_word_vector(option))
#         distances = [cosine_similarity(word_vector.reshape(1,-1), option_vector.reshape(1,-1)) for option_vector in option_vectors]
#         sorted_options = [x for x in sorted(zip(distances, options))]
#         sorted_options = [x[1] for x in sorted(zip(distances, options), reverse=True)]
#         if max_num_outputs == -1:
#             return sorted_options
#         print('Guesses: {}'.format(sorted_options[:max_num_outputs]))
#         return sorted_options[:max_num_outputs]
#
#     def get_embedding_by_index(self, idx):
#         return self.embedding_weights[idx]
#
#     def embed_words_list(self, words_list):
#         word_vectors = []
#         for pos_word in words_list:
#             v = self.get_word_vector(pos_word)
#             if v is not None:
#                 word_vectors.append(v)
#
#         if len(word_vectors) == 0:
#             return None
#
#         return np.asarray(word_vectors)
#

'''gensim version'''

class EmbeddingHandler:
    """
    Parameters
    ----------
    embedding_file : `str`
        Location of a text file containing embeddings in word2vec format.
    """
    def __init__(self, embedding_file: str) -> None:
        # str -> int
        #self.word_indices = {}
        # int -> str
        self.model = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
        self.embedding_weights = self.model.vectors
        self.vocab = self.model.vocab

    def get_word_vector(self, word: str) -> np.ndarray:
        word = word.lower()
        if word not in self.model:
            return None
        return self.model.get_vector(word)

    def sort_options_by_similarity(self,
                                   clue: str,
                                   options: List[str],
                                   max_num_outputs: int = -1) -> List[str]:
        """
        Takes a clue and returns `max_num_outputs` number of `options` sorted by similarity with
        `clue`. If `max_num_outputs` is -1, returns all the options sorted by similarity.
        """
        word = clue.lower()
        if word not in self.model:
            return []
        word_vector = self.get_word_vector(word)
        option_vectors = []
        for option in options:
            if option in self.model:
                option_vectors.append(self.get_word_vector(option))
        distances = [cosine_similarity(word_vector.reshape(1,-1), option_vector.reshape(1,-1)) for option_vector in option_vectors]
        print(sorted(zip(distances,options),reverse=True))
        sorted_options = [x[1] for x in sorted(zip(distances, options), reverse=True)]
        if max_num_outputs == -1:
            return sorted_options
        print('Guesses: {}'.format(sorted_options[:max_num_outputs]))
        return sorted_options[:max_num_outputs]

    def get_embedding_by_index(self, idx):
        return self.embedding_weights[idx]

    def embed_words_list(self, words_list):
        word_vectors = []
        for pos_word in words_list:
            v = self.get_word_vector(pos_word)
            if v is not None:
                word_vectors.append(v)

        if len(word_vectors) == 0:
            return None

        return np.asarray(word_vectors)

    def index_to_word(self, idx):
        return self.model.index2word[idx]