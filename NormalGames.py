

# To-do
# Write an evolutionary optimization algorithm
# Write a multi-modal optimization
# Develop easy to follow outputs
# Add multi-thread workflows (use SCOOP)
# Add docstrings


import numpy as np
import warnings
import math


class PrisonersDilemma(object):
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

    payoffs_matrix.flags.writeable = False

    def __init__(self):
        pass

    def compute_payoffs(self, pl_moves):
        # pl_moves assumed to be ordered dictionary
        # Simple check in the beginnign
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


class DecisionSet(object):
    pass


class DiscreteDecisionSet(DecisionSet):
    # decision_set is coded as discrete set of items to choose
    def __init__(self, decision_set):
        self.type = "Discrete"
        self.decision_set = decision_set
        self.init_strategy()

    ##
    # This block of code is dedicated to handling strategy
    ##

    # Internal method to normalize vectors (used to normalize discr PDF)
    def normalize_strategy(self, vector):
        return vector / np.sum(vector)

    # Defining strategy property
    # Strategy is defined as a probability to select each item of decision_set
    @property
    def strategy(self):
        output = {}
        for idx, mv in enumerate(self.decision_set):
            output[mv] = self._strategy[idx]
        return output

    # This way we change setter method for strategy
    @strategy.setter
    def strategy(self, value):
        # Check if length of strategy vector consistent with decision_set
        # If it's not - raise a ValueError
        if len(value) == len(self.decision_set):
            self._strategy = np.array(value, dtype="float64")
        else:
            raise ValueError("strat length {} not equal DS length {}".format(
                len(value), len(self.decision_set)
            ))

        # Check if strategy vector (which is discrete PDF) is normalized
        # If it's not - raise a warning and normalize
        if not math.isclose(np.sum(self._strategy), 1):
            warnings.warn("Strategy on discrete set is not normalized")
            self._strategy = self.normalize_strategy(self._strategy)

    # Funciton to initialize strategy
    def init_strategy(self):
        self.strategy = self.normalize_strategy(
            np.random.uniform(0, 1, len(self.decision_set)))

    # Function to derive a move for this DecisionSet
    def make_decision(self):
        choice = np.random.choice(self.decision_set, p=self._strategy)
        return choice


class CompactDecisionSet(DecisionSet):
    # Currently only 1-d compacts are supported
    # bounds coded as two items in iterable lower and upper bound
    def __init__(self, bounds):
        self.type = "Compact"
        self.bound_lower, self.bound_upper = bounds
        self.init_strategy()

    # Currently Probability Distribution on compacts are not supported
    # So strategy (Prob.Distr. args) = pure strategy from compact
    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        if self.bound_lower <= value <= self.bound_upper:
            self._strategy = value
        elif value < self.bound_lower:
            warnings.warn("Value for strategy on compact is out of bound")
            self._strategy = self.bound_lower
        elif self.bound_upper < value:
            warnings.warn("Value for strategy on compact is out of bound")
            self._strategy = self.bound_upper

    def init_strategy(self):
        self.strategy = np.random.unif(self.bound_lower,
                                       self.bound_upper, 1)
        return self.strategy

    def make_decision(self):
        return self.strategy


class Player(object):

    """A class of Player object.

    Player class includes tools and methods to operate with sets of
    decision sets binded togethere under an umbrella of a player type.
    """

    # decision_space expected to be an iterable object
    # Each element of decision_space expected to be an instance of DecisionSet
    def __init__(self, name, decision_space=[]):
        self.name = name
        self.decision_space = decision_space

    # It might be useful in future if there would be no init_strategy in DSes
    def init_strategy(self):
        pass

    def make_decision(self):
        """Produce a collective strategies outcome
        on all DSes of the player."""
        move = []
        for ds in self.decision_space:
            move.append(ds.make_decision())
        return move

    def get_strategy(self):
        """Collect and show strategies of the player."""
        strategy = []
        for ds in self.decision_space:
            strategy.append(ds.strategy)
        return strategy


class AgentSet(object):
    """Class to represent a group of players appropriate for the model."""

    def __init__(self, players=[]):
        self.players = players

    def add_player(self, player):
        """Add player to the agent set.

        Check for input provided being a Player class instance.

        """

        if not isinstance(player, Player):
            raise ValueError("player is not a Player instance")
        self.players.append(player)

    def make_move(self):
        """Generate and gather moves from included in the set players.

        Supposed to be random each time called.

        """
        move = {}
        for pl in self.players:
            move[pl.name] = pl.make_decision()
        return move

    def get_strategy(self):
        """Gather strategies from included in the set players."""
        strategy = {}
        for pl in self.players:
            strategy[pl.name] = pl.get_strategy()
        return strategy

