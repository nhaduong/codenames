import operator
import numpy as np
import logging
from itertools import combinations, chain
from nltk.stem import WordNetLemmatizer as wl
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity


from embedding_handler import EmbeddingHandler
from codenames.codenames.utils.game_utils import Clue, DEFAULT_NUM_CLUES, UNREVEALED, GOOD, BAD, CIVILIAN, ASSASSIN, DEFAULT_NUM_TARGETS, CIVILIAN_PENALTY, ASSASSIN_PENALTY, MULTIGROUP_PENALTY, DEPTH, DEFAULT_NUM_GUESSES
from codenames.codenames.guessers.heuristic_guesser import HeuristicGuesser

# guesser_embeddings = EmbeddingHandler('D:/Documents/mycodenames/codenames/data/glove.6B.300d.word2vec.txt')
# giver_embeddings = EmbeddingHandler('D:/Documents/mycodenames/codenames/data/googlenews_SLIM.txt')
guesser_embeddings = EmbeddingHandler('D:/Documents/mycodenames/codenames/data/uk_embeddings.txt')
giver_embeddings = EmbeddingHandler('D:/Documents/mycodenames/codenames/data/uk_embeddings.txt')


def _check_illegal(clue_word, illegal):
    for word in illegal:
        if word in clue_word:
            return True

    return False

def _get_clues(allpos, pos_words_subset, num_clues=75):
    clues = []
    count = len(pos_words_subset) - 1
    lemmatizer = wl()
    stemmer = SnowballStemmer('english')
    illegal = [lemmatizer.lemmatize(word) for word in allpos] + \
              ['indian', 'shakespearean', 'mexican', 'superstar', 'charged', 'theatre', 'theater',
               'duck-billed', 'jay', 'changing', 'charging', 'dancer', 'recheck', 'starlet',
               'dancing', 'operatic', 'beachfront', 'chocolaty', 'typecasting',
               'catfish', 'blueberry', 'raspberry', 'cellular', 'marchers',
               'padlock', 'seaport', 'swung', 'racehorse', 'choco', 'upfield', 'cyclic',
               'racetrack', 'beaten', 'agency', 'german', 'french', 'novella', 'novelist', 'novelette',
               'dino', 'forearm', 'underpants']

    pos_words_vectors = giver_embeddings.embed_words_list(pos_words_subset)

    mean_vector = np.mean(pos_words_vectors, axis=0)
    dotproducts = cosine_similarity(giver_embeddings.embedding_weights, mean_vector.reshape(1, -1)).flatten()

    closest = np.argsort(dotproducts)[::-1]

    for i in range(num_clues):
        clue_index = closest[i]
        clue_word = giver_embeddings.index_to_word(clue_index).lower()
        clue_vector = giver_embeddings.get_embedding_by_index(clue_index)

        if _check_illegal(clue_word, illegal):
            continue
        if clue_word in allpos or lemmatizer.lemmatize(clue_word) in illegal or stemmer.stem(clue_word) in illegal:
            # print('illegal clue: {}'.format(clue_word))
            continue
        if guesser_embeddings.get_word_vector(clue_word) is None:
            continue

        clue_pos_words_similarities = cosine_similarity(pos_words_vectors, clue_vector.reshape(1, -1))

        clue = Clue(clue_word, pos_words_subset, count)
        clues.append((clue, np.max(clue_pos_words_similarities)))
    return clues


def get_clue(board):
    '''dummy version of heuristic_giver which just takes board and finds best clue given the combo of words, assumes no assassin etc'''
    clues_by_group = []
    for group in combinations(range(len(board)), 3):
        target_group = [board[i] for i in group]
        clues_for_group = _get_clues(board, target_group)

        if clues_for_group:
            clues_by_group.append(clues_for_group)

    clues_by_group = list(chain.from_iterable(clues_by_group))
    clues_by_group.sort(key=operator.itemgetter(1), reverse=True)
    clues_by_group = [clue[0] for clue in clues_by_group]
    return clues_by_group[0].clue_word

def samp(vocab, length=5):
    l = np.random.choice(vocab, length, replace=False)
    board = [i for i in l]
    board.append(get_clue(board))
    return board

def make_data():
    with open('D:\Documents\mycodenames\codenames\codenames\gameplay\words.txt') as v:
        vocab = v.readlines()
        vocab = [word.lower().strip('\n') for word in vocab]
        vocab = [word for word in vocab if guesser_embeddings.get_word_vector(word) is not None and giver_embeddings.get_word_vector(word) is not None]

        t = ['train', 'dev', 'test']
        c = [100000, 100, 100]
        sect = []
        for a,b in zip(c,t):
            for i in range(a):
                with open('D:/Documents/mycodenames/codenames/data/ml/'+str(b), 'a+', encoding='utf-8') as f:
                    f.write(','.join(samp(vocab)) + '\n')

def guess_part(file):
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        board_state = [-1] * 5
        boards = [line.split(',')[:-1] for line in lines]
        clues = [line.split(',')[-1].strip('\n') for line in lines]
        # print(lines[:4])
        # clue_vectors = [eh.get_word_vector(clue) for clue in clues]
        guesses = []
        for board, clue in zip(boards, clues):
            g = HeuristicGuesser.guess(board, clue, 3, board_state, 0)
            logging.info('board: {}, clue: {}, guesses: {}'.format(board, clue, g))
            guesses.append(g)
        for guess_group in guesses:
            clues_per_guess_group = get_clue(guess_group)
            logging.info('clue: {}, guess_group: {}'.format(clues_per_guess_group, guess_group))

        with open('first_round', 'a+', encoding='utf-8') as fi:
            for clue, board in clues_per_guess_group, boards:
                new_board_and_clue = ','.join(board) + clue + '\n'
                fi.write(new_board_and_clue)
guess_part('codenames/data/ml/train')
