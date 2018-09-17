from typing import List

from codenames.guessers.guesser import Guesser
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils import game_utils as util

import itertools

THRESHOLD = .02

class PragmaticGuesser(Guesser):
    def __init__(self,
                 embedding_handler: EmbeddingHandler):
        self.embedding_handler = embedding_handler

    def guess(self,
              board: List[str],
              clue: str,
              count: int,
              game_state: List[int],
              current_score: int) -> List[str]:

        #get best options from clue_word to all words on board and sort by descending order
        available_options = util.get_available_choices(board, game_state)
        word = clue.lower()
        if word not in self.embedding_handler.model:
            return []
        word_vector = self.embedding_handler.get_word_vector(word)
        option_vectors = []
        for option in available_options:
            if option in self.model:
                option_vectors.append(self.get_word_vector(option))
        distances = [cosine_similarity(word_vector.reshape(1, -1), option_vector.reshape(1, -1)) for option_vector in
                     option_vectors]
        sorted_options = [x for x in sorted(zip(distances, options), reverse=True)]
        sorted_options = sorted_options[:6]

        #check if similarity scores between close words are within threshold range and should be checked "pragmatically"
        candidates = []
        for pairs in itertools.combinations(range(len(sorted_options)), 2):
            first = sorted_options[pairs[0]]
            second = sorted_options[pairs[1]]
            if first[0] - second[0] < THRESHOLD:
                candidates.append(first)
                candidates.append(second)

        #once we have candidates, see if we can find a word in our own vocabulary that would have made better sense as a clue by querying our vocab for a word that is close to the mean of the 3 words

        return guesses