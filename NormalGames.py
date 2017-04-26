

# To-do
# Write an evolutionary optimization algorithm
# Write a multi-modal optimization
# Develop easy to follow outputs
# Add multi-thread workflows (use SCOOP)
# Add docstrings


import numpy as np
import warnings
import math
import copy
import textwrap
from collections import OrderedDict

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

    def calculate_expected_payoffs(self, pl1_strategy, pl2_strategy):
        payoff = []
        pl1_strategy = np.array(pl1_strategy).reshape(1, -1)
        pl2_strategy = np.array(pl2_strategy).reshape(1, -1)
        pl1_payoff = reduce(np.dot, [pl1_strategy, self.payoffs_matrix[0], pl2_strategy.T])
        pl1_payoff.shape = (1,)
        pl2_payoff = reduce(np.dot, [pl1_strategy, self.payoffs_matrix[1], pl2_strategy.T])
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


class DecisionSet(object):
    def __init__(self):
        self.pl_type = None
        self.type = "Basic"
        self.fitness = np.nan
        self.fitness_averaged = False
        self.evals = None

    def set_pl_type(self, player):
        self.pl_type = player.name


class DiscreteDecisionSet(DecisionSet):
    # decision_set is coded as discrete set of items to choose
    def __init__(self, decision_set, strategy=None):
        """Initialize DDS.

        Parameters
        ----------
        decision_set : iterable
            Iterable of strings with names of possible decisions in the set.
        strategy : list, optional
            List of strategies over decision_set.
            len('strategy') should equal to len('decision_set')

        """
        super().__init__()
        self.type = "DDS"
        self.decision_set = decision_set
        if strategy:
            self.strategy = strategy
        else:
            self.init_strategy()

    ##
    # This block of code is dedicated to handling strategy
    ##

    # Internal method to normalize vectors (used to normalize discr PDF)
    def normalize_strategy(self, vector):
        if np.allclose(vector, np.zeros(len(vector))):
            vector = np.ones(len(vector))

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
            warnings.warn(
                "strat {} on DDS not normalized".format(self._strategy))
            self._strategy = self.normalize_strategy(self._strategy)

    # Funciton to initialize strategy
    def init_strategy(self):
        self.strategy = self.normalize_strategy(
            np.random.uniform(0, 1, len(self.decision_set)))

    # Function to derive a move for this DecisionSet
    def make_decision(self):
        choice = np.random.choice(self.decision_set, p=self._strategy)
        return choice

    def mutate(self, magnitude):
        """Simple wrapper to internal mutate method."""
        self.fitness = np.nan
        self.fitness_averaged = False
        self.evals = None
        self._mutate_all(magnitude)

    def _mutate_all(self, magnitude):
        """Mutate a DDS without output."""
        if not math.isclose(magnitude, 0):
            strat_candidates = self._strategy + np.random.normal(0, magnitude,
                                                                 len(self._strategy))
        else:
            strat_candidates = self._strategy
        strat_candidates = np.clip(strat_candidates, 0, 1)
        self.strategy = self.normalize_strategy(strat_candidates)

    def _mutate_opt(self, magnitude):
        """Mutate a DDS without output."""

        coded_strat = self._strategy[:-1]

        strat_candidates = coded_strat + np.random.normal(0, magnitude,
                                                          len(coded_strat))
        # print(type(strat_candidates), strat_candidates)

        strat_candidates = np.clip(strat_candidates, 0, 1)
        # print(type(strat_candidates), strat_candidates)

        strat_candidates = np.concatenate((strat_candidates,
                                           1 - np.sum(strat_candidates,
                                                      keepdims=True)))

        strat_candidates[-1] = np.clip(strat_candidates[-1], 0, 1)
        self.strategy = self.normalize_strategy(strat_candidates)

    def __repr__(self):
        output = []
        output.append(
            "DDS / {} / {} choices:".format(self.pl_type, len(self.decision_set)))
        output.append(repr(self.strategy))
        return ' '.join(output)


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

    Parameters
    ----------
    name : string
        Player type
    deicison_space : list
        List of instances of DeicisionSet class
    """

    # decision_space expected to be an iterable object
    # Each element of decision_space expected to be an instance of DecisionSet
    def __init__(self, name, decision_space):
        self.name = name
        self.decision_space = decision_space
        for ds in decision_space:
            ds.set_pl_type(self)

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

    def __repr__(self):
        output = []
        output.append("{} / {} decision sets:".format(self.name,
                                                      len(self.decision_space)))
        for ds in self.decision_space:
            output.append(textwrap.indent(repr(ds), "    "))
        output.append("")
        return "\n".join(output)


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

    # Useful representations for simplicity of use ---------------------------

    def __len__(self):
        return len(self.players)

    def __repr__(self):
        output = []
        output.append("AgentSet / {} agents:".format(len(self.players)))
        for pl in self.players:
            output.append(textwrap.indent(repr(pl), '  '))
        return "\n".join(output)


class SubPopulation(object):
    """SubPopulation abstraction consisting of the same decision_set"""

    def __init__(self, decision_set, popsize):
        self._init_ds = copy.deepcopy(decision_set)
        self._popsize = popsize
        self._population = []
        self._population.append(copy.deepcopy(decision_set))
        self.pl_type = decision_set.pl_type

    def add_member(self, ds):
        if not isinstance(ds, DecisionSet):
            raise ValueError("subp new member should of DecisionSet class")
        if ds.decision_set != self._init_ds.decision_set:
            raise ValueError("new member type doesn't match subp type")
        if len(self._population) + 1 > self._popsize:
            raise OverflowError("cannot add new member: subp size overflow")

        self._population.append(copy.deepcopy(ds))

    def unfold_population(self):
        for i in range(len(self._population), self._popsize):
            self.add_member(self[0])

    def mutate(self, *args, **kwargs):
        for ds in self:
            ds.mutate(*args, **kwargs)

    def __len__(self):
        return len(self._population)

    def __getitem__(self, key):
        if isinstance(key, slice):
            output = copy.copy(self)
            output._population = self._population[key]
            return output
        return self._population[key]

    def __repr__(self):
        output = []
        output.append("Subpop / size {} of {} / {} / {}".format(len(self),
                                                                self._popsize, self.pl_type, self._init_ds.decision_set))
        for ds in self._population:
            output.append(textwrap.indent(repr(ds), "  "))

        return "\n".join(output)


class Population(object):
    """To-do: Aggregation of SubPopulations to manage them in tractable way."""

    def __init__(self, agentset=AgentSet(), popsize=20, mutation_magnitude=0.03):
        self.agentset = agentset
        self._pl_count = len(self.agentset)
        self._population = []
        self._mt_magnitude = mutation_magnitude
        self._popsize = popsize
        self.init_population()

    def init_population(self):
        """Create a dcitionary with each ds of each player putted under different dictionary key."""

        for pl in self.agentset.players:
            for ds in pl.decision_space:
                if ds.pl_type != pl.name:
                    warnings.warn("pl_type of\n{} of {} not in sync".format(ds, pl))
                    ds.set_pl_type(pl)

                self._population.append(SubPopulation(decision_set=ds,
                                                      popsize=self._popsize))
        self.unfold_population()

    def unfold_population(self):
        """"Extend popTable provided by makePopulationTable
        method to the size specified by _popsize attribute."""

        for subp in self:
            subp.unfold_population()

    def mutate(self, *args, **kwargs):
        """Mutate every DS in the population
        calling mutate method on them."""
        # print(self._population)
        for subp in self:
            subp.mutate(*args, **kwargs)
        # print(self._population)

    def __getitem__(self, key):
        if isinstance(key, slice):
            output = copy.copy(self)
            output._population = self._population[key]
            return output
        return self._population[key]

    def __repr__(self):
        output = []
        output.append(
            "Population : (size = {} / subpopulations = {})".format(self._popsize, len(self._population)))

        for subp in self._population:
            # output.append("  Subpop {}".format(ds_at))
            # for ds in subp:
            output.append(textwrap.indent(repr(subp), "  "))

        return "\n".join(output)


class EvolutionaryEquilibrium(object):
    """Utils to deal with evolutionary search for equilibrium states."""

    def __init__(self, agentset=None, model=None,
                 popsize=6, generations=10,
                 mutation_magnitude=0.03, npairs=5, ngames=2):
        self.agentset = copy.deepcopy(agentset)
        self.model = copy.deepcopy(model)
        self._mt_magnitude = mutation_magnitude
        self._npairs = npairs
        self._ngames = ngames
        self._popsize = popsize
        self.pop = Population(agentset=self.agentset,
                              popsize=self._popsize,
                              mutation_magnitude=self._mt_magnitude)

    @staticmethod
    def make_agentset(population, agentset):
        if not isinstance(population, Population):
            raise ValueError("population must be inst of Population()")
        if not isinstance(agentset, AgentSet):
            raise ValueError("agentset must be inst of AgentSet()")


        # ref_players_dict = {pl.name: pl for pl in agentset.players}
        output_players = OrderedDict()

        for pl in agentset.players:
            output_players[pl.name] = Player(pl.name, [])

        for subp in population:
            output_players[subp.pl_type].decision_space.append(subp[0])
        
        output = AgentSet(output_players.values())

        return output

    # dummy funciton to be written
    def calculate_fitness(self, npairs=self._npairs, ngames=self._ngames):
        # matchings = self.make_matchings(npairs)
        # for matching in matchings:
        #     self.calculate_payoffs(matching, ngames)
        pass

    # dummy function to be written
    def make_matchings(self, npairs):
        # for subp in self.pop:
        pass

    def __repr__(self):
        output = []
        output.append("EvolutionaryEquilibrium(popsize={}, mutation_magnitude={})\n".format(self._popsize,
                                                                                            self._mt_magnitude))
        # output.append("")
        output.append("AgentSet:")
        output.append("---------")
        output.append(repr(self.agentset))
        output.append("Population:")
        output.append("---------")
        output.append(repr(self.pop))
        return "\n".join(output)

        pass
