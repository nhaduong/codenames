from typing import List

from overrides import overrides

from codenames.embedding_handler import EmbeddingHandler
from codenames.guessers.guesser import Guesser
import codenames.utils.game_utils as util


class HeuristicGuesser(Guesser):

    def __init__(self, board: List[str], embedding_handler: EmbeddingHandler):
        super().__init__(board)
        self.embedding_handler = embedding_handler

    @overrides
    def guess(self,
              clue: str,
              count: int,
              game_state: List[int],
              current_score: int) -> List[str]:
        available_options = util.get_available_choices(self.board, game_state)
        # Return 1 more than the count because the game rules allow it.
        return self.embedding_handler.sort_options_by_similarity(clue,
                                                                 available_options,
                                                                 count + 1)
