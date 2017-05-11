#
#
# This file contains a collection of games for EVGT
#
#

import numpy as np


class Game(object):
    """docstring for Game"""

    def __init__(self, arg):
        pass


class NormalFormGame(Game):
    """Basic PrisonersDilemma game 2x2"""
    players_count = 0
    payoffs_matrix = np.array(
        [[[0, 0],
          [0, 0]],

         [[0, 0],
          [0, 0]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Strategy_1": 0, "Strategy_2": 1}
    allowed_moves["Player_2"] = {"Strategy_1": 0, "Strategy_2": 1}

    # payoffs_matrix.flags.writeable = False

    def __init__(self):
        pass

    def calculate_payoffs(self, pl_moves):
        """Calculate payoffs for provided moves.

        Input
        -----
        pl_moves : dictionary
            Dictionary with pl_types (names) as keys
            and list of their actions as values of the dictionary

        Output
        ------
        dictionary :
            Payoffs of players. Dictionary with
            the same keys as input dictionary and integer payoffs
            as values of the dictionary.

        """

        # Check that the length of provided vector of strategies
        # equals the number of specified in the game
        if len(pl_moves) != self.players_count:
            raise Exception("Pure strategies set length is not compatible")

        # pl_moves_vector gathering moves coded in tractable way
        pl_moves_vector = []

        for pl, strat in pl_moves.items():
            # Checking if provided player type allowed by the model
            if pl not in self.players:
                raise Exception("player type '{}' not allowed".format(
                    pl
                ))

            # Checking if provided move could be handled by the model
            if strat[0] not in self.allowed_moves[pl]:
                raise Exception("Move '{}' for '{}' not allowed".format(
                    strat[0], pl
                ))

            pl_moves_vector.append(self.allowed_moves[pl][strat[0]])

        # Deriving vector with players payoffs
        indexing_for_payoffs_vector = (slice(None),) + tuple(pl_moves_vector)
        payoffs_vector = self.payoffs_matrix[indexing_for_payoffs_vector]

        # Generating a dictionary with payoffs symmetrical to pl_moves
        payoffs = {}
        for index, player in enumerate(pl_moves.keys()):
            payoffs[player] = payoffs_vector[index]

        return payoffs

    def calculate_expected_payoffs(self, pl1_strategy, pl2_strategy):
        """Calculate expected payoffs for provided strategies.

        !Supported only for 2x2 games

        """
        pl1_strategy = np.array(pl1_strategy).reshape(1, -1)
        pl2_strategy = np.array(pl2_strategy).reshape(1, -1)
        pl1_payoff = reduce(
            np.dot, [
                pl1_strategy, self.payoffs_matrix[0], pl2_strategy.T])
        pl1_payoff.shape = (1,)
        pl2_payoff = reduce(
            np.dot, [
                pl1_strategy, self.payoffs_matrix[1], pl2_strategy.T])
        pl2_payoff.shape = (1,)
        return pl1_payoff, pl2_payoff


class PrisonersDilemma(NormalFormGame):
    """Basic PrisonersDilemma game 2x2"""
    players_count = 2
    payoffs_matrix = np.array(
        [[[-1, -3],
          [0, -2]],

         [[-1, 0],
          [-3, -2]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Cooperate": 0, "Defect": 1}
    allowed_moves["Player_2"] = {"Cooperate": 0, "Defect": 1}

    # payoffs_matrix.flags.writeable = False

    def __init__(self):
        pass


class StagHunt(NormalFormGame):
    payoffs_matrix = np.array(
        [[[10, 0],
          [8, 7]],

            [[10, 8],
             [0, 7]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Stag": 0, "Hunt": 1}
    allowed_moves["Player_2"] = {"Stag": 0, "Hunt": 1}

    def __init__(self):
        pass


class OnlyMixed2x2(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[4, 11],
          [8, 10]],

            [[3, 0],
             [0, 2]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Strategy_1": 0, "Strategy_2": 1}
    allowed_moves["Player_2"] = {"Strategy_1": 0, "Strategy_2": 1}

    def __init__(self):
        pass


class OnlyOne2x2(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[4, 0],
          [-1, 2]],

            [[1, 0],
             [0, 2]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Strategy_1": 0, "Strategy_2": 1}
    allowed_moves["Player_2"] = {"Strategy_1": 0, "Strategy_2": 1}

    def __init__(self):
        pass


class ChoosingSides(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[10, 0],
          [0, 10]],

            [[10, 0],
             [0, 10]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Left": 0, "Right": 1}
    allowed_moves["Player_2"] = {"Left": 0, "Right": 1}

    def __init__(self):
        pass


class BoS(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[4, 0, 0],
          [0, 2, 1]],

         [[2, 0, 1],
          [0, 4, 3]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"B": 0, "S": 1}
    allowed_moves["Player_2"] = {"B": 0, "S": 1, "X": 0}

    def __init__(self):
        pass


class RockPaperScissors(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[0, -1, 1],
          [1, 0, -1],
          [-1, 1, 0]],

         [[0, 1, -1],
          [-1, 0, 1],
          [1, -1, 0]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"R": 0, "P": 1, "S": 2}
    allowed_moves["Player_2"] = {"R": 0, "P": 1, "S": 2}

    def __init__(self):
        pass


class Chicken(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[0, 7],
          [2, 6]],

            [[0, 2],
             [7, 6]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Dare": 0, "Chicken": 1}
    allowed_moves["Player_2"] = {"Dare": 0, "Chicken": 1}

    def __init__(self):
        pass


class SwerveStay(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[0, -1],
          [1, -20]],

            [[0, 1],
             [-1, -20]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Swerve": 0, "Stay": 1}
    allowed_moves["Player_2"] = {"Swerve": 0, "Stay": 1}

    def __init__(self):
        pass


class HawkDove(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[-5, 10],
          [0, 5]],

            [[-5, 0],
             [10, 5]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Hawk": 0, "Dove": 1}
    allowed_moves["Player_2"] = {"Hawk": 0, "Dove": 1}

    def __init__(self):
        pass


class HawkDoveSkewedDove(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[-7.5, 5],
          [0, 2.5]],

            [[-7.5, 0],
             [5, 2.5]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Hawk": 0, "Dove": 1}
    allowed_moves["Player_2"] = {"Hawk": 0, "Dove": 1}

    def __init__(self):
        pass


class HawkDoveSkewedHawk(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[-2.5, 15],
          [0, 7.5]],

            [[-2.5, 0],
             [15, 7.5]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"Hawk": 0, "Dove": 1}
    allowed_moves["Player_2"] = {"Hawk": 0, "Dove": 1}

    def __init__(self):
        pass


class HarmThyNeighbour(NormalFormGame):
    players_count = 2
    payoffs_matrix = np.array(
        [[[2, 1],
          [2, 2]],

            [[2, 2],
             [1, 2]]])
    players = ["Player_1", "Player_2"]
    allowed_moves = {}
    allowed_moves["Player_1"] = {"A": 0, "B": 1}
    allowed_moves["Player_2"] = {"A": 0, "B": 1}

    def __init__(self):
        pass
