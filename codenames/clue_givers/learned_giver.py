import operator
from itertools import combinations, chain, permutations
from typing import List
import time
from overrides import overrides

import codenames.utils.game_utils as util
import numpy as np
from random import choices
from nltk.stem import WordNetLemmatizer as wl
from nltk.stem.snowball import SnowballStemmer
from sklearn.metrics.pairwise import cosine_similarity

from codenames.clue_givers.giver import Giver
from codenames.embedding_handler import EmbeddingHandler
from codenames.utils.game_utils import Clue, DEFAULT_NUM_CLUES, UNREVEALED, GOOD, BAD, CIVILIAN, ASSASSIN, DEFAULT_NUM_TARGETS, CIVILIAN_PENALTY, ASSASSIN_PENALTY, MULTIGROUP_PENALTY, DEPTH, DEFAULT_NUM_GUESSES


LEARNING_RATE = .005

import numpy as np

class LearnedGiver(Giver):
    def __init__(self,
                 embedding_handler: EmbeddingHandler,
                 guesser_embedding_handler: EmbeddingHandler,
                 blacklist: List[str] = [],
                 current_board: List[str] = None,
                 train: bool=False) -> None:

        self.embedding_handler = embedding_handler
        self.guesser_embedding_handler = guesser_embedding_handler
        self.blacklist = blacklist
        self.current_board = current_board

        self.train = train
        self.projection = np.add(np.random.normal(0,.1,embedding_handler.embedding_weights.shape[0]),np.identity(embedding_handler.embedding_weights.shape[0])) #initialize identity matrix with some noise
        self.projection_history = []
        self.projection_log_probs = []

    def _update_embeddings(self,
                           embedding_handler: EmbeddingHandler,
                           projection: List[float]) -> EmbeddingHandler:
        return np.matmul(projection, embedding_handler.embedding_weights)

    def _check_illegal(self,
                       clue_word: str,
                       illegal: List[str]) -> bool:
        for word in illegal:
            if word in clue_word:
                return True
            else:
                return False
    def sim2prob(self,
                 clue_word: str,
                 board: List[str],
                 embedding_handler: EmbeddingHandler) -> List[float]:
        '''Converts 1 - similarity scores into probabilities. Compute on 1-sim because we want the lowest similarity to have the highest distance.
            Converts by normalizing inverse distances with: 1/dist / sum(1/dist). Could possibly try using squares or higher order inverses to skew probability
            distribution more. Higher skew might make it easier to learn?
            outputs shape (len(board), 1, 1)
        '''

        clue_neg_sims = np.array([1-cosine_similarity(embedding_handler.get_word_vector(board_word).reshape(1,-1), embedding_handler.get_word_vector(clue_word).reshape(1,-1)) for board_word in board])
        # print(clue_neg_sims)
        # return self.stable_softmax(clue_neg_sims)
        sim_sum = np.sum(1/clue_neg_sims)
        clue_neg_probs = [1/sim / sim_sum for sim in clue_neg_sims]
        #assert np.sum(clue_neg_probs) == 1, 'Distances converted to probabilities don\'t add up to 1!, {}{}'.format(np.sum(clue_neg_probs), clue_neg_probs)
        return clue_neg_probs

    ''' Returns list of n=NUM_CLUES of Clues for a given group of words'''
    def _get_clues(self, allpos, pos_words_subset, neg_words, civ_words, ass_words, available_illegal, aggressive, random_clues=False, num_clues=DEFAULT_NUM_CLUES,MULTIGROUP_PENALTY=MULTIGROUP_PENALTY):
        clues = []
        count = len(pos_words_subset) - 1
        lemmatizer = wl()
        stemmer = SnowballStemmer('english')
        allpos_vectors = self.embedding_handler.embed_words_list(allpos)
        illegal = [lemmatizer.lemmatize(word) for word in available_illegal] + \
                  available_illegal + \
                  ['indian', 'shakespearean', 'mexican', 'superstar', 'charged', 'theatre', 'theater',
                   'duck-billed', 'jay', 'changing', 'charging', 'dancer', 'recheck', 'starlet',
                   'dancing','operatic','beachfront','chocolaty','typecasting',
                   'catfish','blueberry','raspberry','cellular','marchers',
                   'padlock','seaport','swung','racehorse','choco','upfield','cyclic',
                   'racetrack','beaten','agency','german','french','novella','novelist','novelette',
                   'dino','forearm','underpants']
        pos_words_vectors = self.embedding_handler.embed_words_list(pos_words_subset)
        neg_words_vectors = self.embedding_handler.embed_words_list(neg_words)
        civ_words_vectors = self.embedding_handler.embed_words_list(civ_words)
        ass_words_vectors = self.embedding_handler.embed_words_list(ass_words)
        if pos_words_vectors is None or neg_words_vectors is None:
            return None

        mean_vector = np.mean(pos_words_vectors,axis=0)
        dotproducts = cosine_similarity(self.embedding_handler.embedding_weights, mean_vector.reshape(1,-1)).flatten()
        #mean_vector /= np.sqrt(mean_vector.dot(mean_vector))
        #dotproducts = np.dot(self.embedding_handler.embedding_weights, mean_vector).reshape(-1)
        closest = np.argsort(dotproducts)[::-1]
        '''Skew 'good clues' towards larger groups of target words'''
        if aggressive:
            if count <= 1:
                MULTIGROUP_PENALTY += 0
            elif count <= 3:
                MULTIGROUP_PENALTY += .0
            else:
                MULTIGROUP_PENALTY += .0
        max_greater = 0
        civ_greater = 0
        ass_greater = 0
        num_clues_tried = 0
        #print('neg: {}, civ: {}, ass: {}, cluecount: {}'.format(max_greater, civ_greater, ass_greater, num_clues_tried))


        for i in range(num_clues):
            num_clues_tried += 1
            clue_index = closest[i]
            clue_word = self.embedding_handler.index_to_word(clue_index).lower()
            clue_vector = self.embedding_handler.get_embedding_by_index(clue_index)

            if self._check_illegal(clue_word,illegal):
                continue
            if clue_word in allpos or lemmatizer.lemmatize(clue_word) in illegal or stemmer.stem(clue_word) in illegal:
                #print('illegal clue: {}'.format(clue_word))
                continue


            allpos_similarities = cosine_similarity(allpos_vectors, clue_vector.reshape(1,-1))
            min_allpos_cosine = np.min(allpos_similarities)
            clue_pos_words_similarities = cosine_similarity(pos_words_vectors, clue_vector.reshape(1,-1))

            if random_clues:
                clue = Clue(clue_word, pos_words_subset, count)
                clues.append((clue,np.max(clue_pos_words_similarities)))
                continue

            clue_neg_words_similarities= cosine_similarity(neg_words_vectors, clue_vector.reshape(1,-1))
            min_clue_cosine = np.min(clue_pos_words_similarities) + MULTIGROUP_PENALTY
            if count > 1: min_clue_cosine += 0

            #logging.info('potential clue : {}'.format(clue_word))

            max_neg_cosine = np.max(clue_neg_words_similarities)
            if max_neg_cosine >= min_clue_cosine:
                max_greater += 1
                continue
            if civ_words_vectors is not None:
                clue_civ_words_similarities = cosine_similarity(civ_words_vectors, clue_vector.reshape(1,-1))
                max_civ_cosine = np.max(clue_civ_words_similarities)
                if max_civ_cosine >= min_clue_cosine + CIVILIAN_PENALTY:
                    civ_greater += 1
                    continue
            if ass_words_vectors is not None:
                max_ass_cosine = cosine_similarity(ass_words_vectors,clue_vector.reshape(1,-1))
                if max_ass_cosine >= min_clue_cosine - ASSASSIN_PENALTY:
                    ass_greater +=1
                    continue
            clue = Clue(clue_word, pos_words_subset, count)
            clues.append((clue,np.max(clue_pos_words_similarities)))
        return clues

    '''List of Clues sorted by descending Cosine distance'''

    def _guess(self,
                        board: List[str],
                        clue: str,
                        count: int,
                        game_state: List[int],
                        ) -> List[str]:
        available_options = util.get_available_choices(board, game_state)
        # Return 1 more than the count because the game rules allow it.
        return self.embedding_handler.sort_options_by_similarity(clue,
                                                                 available_options,
                                                                 count + 1)

    def remove_dups(self,
                    clues: List[tuple]) -> List[tuple]:
        last = object()
        for item in clues:
            if item[0] == last:
                continue
            yield item
            last = item[0]

    def stable_softmax(self, X): #outputs array that almost adds up to 1
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)

    def _skipgram(self,
                  board: List[str],
                  clue: tuple,
                  game_state: List[int]):
        '''returns updated projection of original embedding_handler and loss'''
        giver_truth = [1] * len(board) #predicted
        guesser_truth = [1] * len(board) #real truth

        for word in clue.intended_board_words:
            idx = board.index(word)
            giver_truth[idx] = 0
        guesses = self._guess(board, clue.clue_word, -2, game_state)
        for word in guesses[:3]:
            idx = board.index(word)
            guesser_truth[idx] = 0

        #softmax the sim scores from clueword to board words
        giver_probs = self.sim2prob(clue.clue_word, board, self.embedding_handler)  # ? or keep it as (25,1,1)?
        #guesser_probs = self.sim2prob(clue.clue_word, board, self.guesser_embedding_handler)

        loss = 0
        for y, p in zip(guesser_truth, giver_probs):
            loss += -(y * np.log(p) + (1 - y) * np.log(1 - p))

        error = np.array([-label + g for label, g in zip(guesser_truth, self.stable_softmax(giver_probs)) ])
        dW2 = np.outer(self.projection, np.sum(error, axis=0))
        self.projection -= (LEARNING_RATE * dW2).reshape(self.projection.shape)
        new_embeddings = self._update_embeddings(self.embedding_handler, self.projection)

        self.projection_history.append(self.projection)
        return new_embeddings, loss


    def get_next_clue(self,
                      board: List[str],
                      allIDs: List[int],
                      game_state: List[int],
                      score: int,
                      previous_guess: List[str] = None,
                      ) -> List[Clue]:
        '''Creates empty blacklist upon each new game'''
        if self.current_board != board:
            self.blacklist = []
            self.current_board = board


        pos_words = [board[idx] for idx, val in enumerate(allIDs) if val == GOOD]
        neg_words = [board[idx] for idx, val in enumerate(allIDs) if val == BAD]
        civ_words = [board[idx] for idx, val in enumerate(allIDs) if val == CIVILIAN]
        ass_words = [board[idx] for idx, val in enumerate(allIDs) if val == ASSASSIN]

        available_targets = [word for word in pos_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_neg = [word for word in neg_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_civ = [word for word in civ_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_ass = [word for word in ass_words if
                             game_state[board.index(word)] == UNREVEALED]
        available_illegal = available_targets + available_neg + available_civ + available_ass
        num_revealed = 0
        for idx, value in enumerate(game_state):
            if value != UNREVEALED:
                num_revealed += 1
        if num_revealed > len(game_state) / 2 and score < num_revealed:
            aggressive = False
        else:
            aggressive = False
        if len(available_targets) > DEFAULT_NUM_TARGETS:
            num_words = DEFAULT_NUM_TARGETS
        else:
            num_words = len(available_targets)
        clues_by_group = []

        if len(available_targets) >= 3:
            for group in combinations(range(len(available_targets)),DEFAULT_NUM_TARGETS):
                target_group = [available_targets[i] for i in group]
                clues_for_group = self._get_clues(pos_words, target_group, available_neg, available_civ, available_ass,
                                                  available_illegal, aggressive)

                if clues_for_group:
                    clues_by_group.append(clues_for_group)

        if len(clues_by_group) == 0:
            for count in range(len(available_targets), 0, -1):
                for group in combinations(range(num_words), count):
                    target_group = [available_targets[i] for i in group]
                    clues_for_group = self._get_clues(pos_words, target_group, available_neg, available_civ, available_ass, available_illegal, aggressive)

                    if  clues_for_group:
                        clues_by_group.append(clues_for_group)

        clues_by_group = list(chain.from_iterable(clues_by_group))
        clues_by_group.sort(key=operator.itemgetter(1),reverse=True)
        clues_by_group = [clue[0] for clue in clues_by_group]

        #filter out clues    TODO: make a filter function
        filtered_clues_by_group = [clue for clue in clues_by_group if clue.clue_word not in self.blacklist]
        if filtered_clues_by_group:
            self.blacklist.append(filtered_clues_by_group[0].clue_word)
        while filtered_clues_by_group and self.guesser_embedding_handler.get_word_vector(filtered_clues_by_group[0].clue_word) is None:
            self.blacklist.append(filtered_clues_by_group[0])
            filtered_clues_by_group.pop(0)

        #safety measure for if no good clue can be generated; just gets top N clues that are closest to pos words, ignoring negative words
        if len(filtered_clues_by_group) == 0:
            clues_by_group = []
            for count in range(len(available_targets), 0, -1):
                for group in combinations(range(num_words), count):
                    target_group = [available_targets[i] for i in group]
                    clues_for_group = self._get_clues(pos_words, target_group, available_neg, available_civ, available_ass, available_illegal, aggressive, random_clues=True)
                if clues_for_group:
                    clues_by_group.append(clues_for_group)
            clues_by_group = list(chain.from_iterable(clues_by_group))
            clues_by_group.sort(key=operator.itemgetter(1), reverse=True)
            clues_by_group = [clue[0] for clue in clues_by_group]
            filtered_clues_by_group = [clue for clue in clues_by_group if clue.clue_word not in self.blacklist]
        if filtered_clues_by_group:
            self.blacklist.append(filtered_clues_by_group[0].clue_word)
        while filtered_clues_by_group and self.guesser_embedding_handler.get_word_vector(filtered_clues_by_group[0].clue_word) is None:
            self.blacklist.append(filtered_clues_by_group[0])
            filtered_clues_by_group.pop(0)

        clue = filtered_clues_by_group[0]
        print(self.embedding_handler)
        #update the embedding handler for the next turn
        self.embedding_handler.vocab, loss = self._skipgram(board, clue, game_state)
        print("loss: {}".format(loss))
        #print(self.projection_history)
        #print(self.embedding_handler)
        return filtered_clues_by_group

def main():
    embed = '/home/nikita/Downloads/codenames/data/uk_embeddings.txt'
    test_embed = EmbeddingHandler(embed)
    hg = LearnedGiver(test_embed, EmbeddingHandler('/home/nikita/Downloads/codenames/data/uk_embeddings.txt'))
    board = ['witch','cap','novel','bear','tooth','tennis','scientist','ham','spell','bark','dragon','embassy','row','club','book','fire','pole','green','force','whip','life','canada','tag','nail','mouse']
    allIDs = [ASSASSIN,GOOD,GOOD,GOOD,GOOD,GOOD,GOOD,GOOD,GOOD,GOOD,BAD,BAD,BAD,BAD,BAD,BAD,BAD,BAD,CIVILIAN,CIVILIAN,CIVILIAN,CIVILIAN,CIVILIAN,CIVILIAN,CIVILIAN]
    game_state = [UNREVEALED, UNREVEALED, UNREVEALED, UNREVEALED, 2, UNREVEALED, UNREVEALED, 2,UNREVEALED,UNREVEALED,UNREVEALED,2,2,UNREVEALED,UNREVEALED,UNREVEALED,UNREVEALED,UNREVEALED,UNREVEALED,2,UNREVEALED,UNREVEALED,UNREVEALED,2,UNREVEALED]
    score = 3
    cg = hg.get_next_clue(board,allIDs,game_state,score)
    print(cg[:15])

if __name__ == "__main__":
    main()