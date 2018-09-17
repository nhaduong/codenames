from collections import namedtuple
from typing import List
from collections import namedtuple

UNREVEALED = -1

GOOD = 1

BAD = 2

CIVILIAN = 3

ASSASSIN = 0

Clue = namedtuple('Clue', ['clue_word', 'intended_board_words', 'count'])

DEFAULT_NUM_CLUES = 150

DEFAULT_NUM_TARGETS = 3

CIVILIAN_PENALTY = .0
ASSASSIN_PENALTY = .17
MULTIGROUP_PENALTY = .0
DEPTH = 5
DEFAULT_NUM_GUESSES = 5

def get_available_choices(board: List[str],
                          game_state: List[int]) -> List[str]:
    assert len(board) == len(game_state), "Invalid state!"
    options = []
    for string, state_id in zip(board, game_state):
        if state_id == UNREVEALED:
            options.append(string)
    return options
