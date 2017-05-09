
# To-do
# Write an evolutionary optimization algorithm
# Write a multi-modal optimization
# [done] Develop easy to follow outputs
# Add multi-thread workflows (use SCOOP)
# Add docstrings

import numpy as np
import warnings
import math
import copy
import textwrap
from collections import OrderedDict
import logging
from logging.handlers import MemoryHandler
import random
from operator import attrgetter
# import pandas as pd
# import pdb
import json
import csv
import time
import os
from functools import reduce
from tqdm import tqdm, tqdm_notebook
from concurrent import futures
import uuid


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
        if len(self.decision_set) == 2:
            tmp = np.random.random()
            if tmp < self._strategy[0]:
                choice = self.decision_set[0]
            else:
                choice = self.decision_set[1]
        else:
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
            strat_candidates = self._strategy + np.random.normal(
                0, magnitude, len(self._strategy))
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

    def __str__(self):
        output = []
        output.append(
            "DDS / {} / {} fit / {} choices:".format(self.pl_type,
                                                     self.fitness,
                                                     len(self.decision_set)))
        output.append(repr(self.decision_set))
        return ' '.join(output)

    def __repr__(self):
        output = []
        if self.fitness_averaged:
            avg_sign = "AVG"
        else:
            avg_sign = "RAW"

        output.append(
            "DDS / {} / {:.5f} fit - {} / {} evals:".format(self.pl_type,
                                                            self.fitness,
                                                            avg_sign,
                                                            self.evals))
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
        output.append("{} / {} decision sets:".format(
            self.name, len(self.decision_space)))
        for ds in self.decision_space:
            output.append(textwrap.indent(repr(ds), "    "))
        output.append("")
        return "\n".join(output)


class AgentSet(object):
    """Class to represent a group of players appropriate for the model."""

    def __init__(self, players=[], model=None):
        if model is None:
            self.players = players
        # If appropriate model provided as input, initialize corresponding
        # AgentSet()
        elif isinstance(model, Game):
            self._read_model(model)
        else:
            raise ValueError(
                "Provided model input to AgentSet() is not legible")

    def _read_model(self, model):
        if isinstance(model, NormalFormGame):
            self.players = []
            for pl_type in model.players:
                pl_dds = DiscreteDecisionSet(list(
                    model.allowed_moves[pl_type].keys()))
                pl = Player(pl_type, [pl_dds])
                self.players.append(pl)
        else:
            raise ValueError("Model type provided for AgentSet is unsupported")

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

    def __getitem__(self, key):
        return self.players[key]

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
        self._init_ds = decision_set
        self._popsize = popsize
        self._population = []
        self._population.append(decision_set)
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

    def init_uniform_population(self):
        for i in range(len(self._population), self._popsize):
            logging.debug("Adding new random member to subp: current size {} targeted {}".format(
                len(self._population), self._popsize))
            new_member = copy.copy(self[0])
            new_member.init_strategy()
            self.add_member(new_member)
            logging.debug("Added new member")

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

    def __delitem__(self, key):
        if isinstance(key, slice):
            del(self._population[key])
        else:
            del(self._population[key])

    def __str__(self):
        output = []
        output.append("Subpop / size {} of {} / {} / {}".format(len(self),
                                                                self._popsize, self.pl_type, self._init_ds.decision_set))
        return "\n".join(output)

    def __repr__(self):
        output = []
        output.append("Subpop / size {} of {} / {} / {}".format(len(self),
                                                                self._popsize, self.pl_type, self._init_ds.decision_set))
        for ds in self._population:
            output.append(textwrap.indent(repr(ds), "  "))

        return "\n".join(output)


class Population(object):
    """To-do: Aggregation of SubPopulations to manage them in tractable way."""

    def __init__(self, agentset=AgentSet(), init_uniform=True,
                 popsize=20, mutation_magnitude=0.03):
        self.agentset = agentset
        self._pl_count = len(self.agentset)
        self._population = []
        self._mt_rate = mutation_magnitude
        self._popsize = popsize
        self.init_population(uniform=init_uniform)

    def init_population(self, uniform):
        """Create a dcitionary with each ds of each player putted under different dictionary key."""

        for pl in self.agentset.players:
            for ds in pl.decision_space:
                if ds.pl_type != pl.name:
                    warnings.warn(
                        "pl_type of\n{} of {} not in sync".format(ds, pl))
                    ds.set_pl_type(pl)

                self._population.append(SubPopulation(decision_set=copy.deepcopy(ds),
                                                      popsize=self._popsize))
        if uniform:
            for subp in self:
                subp.init_uniform_population()
        else:
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

    def __setitem__(self, key, value):
        if not isinstance(value, SubPopulation):
            raise ValueError("assigning value must be inst of SubPopulation()")
        self._population[key] = value

    def append(self, value):
        if not isinstance(value, SubPopulation):
            raise ValueError("appending value must be inst of SubPopulation()")
        self._population.append(value)
        pass

    def __repr__(self):
        output = []
        output.append(
            "Population : (size = {} / subpopulations = {})".format(
                self._popsize, len(self._population)))

        for subp in self._population:
            # output.append("  Subpop {}".format(ds_at))
            # for ds in subp:
            output.append(textwrap.indent(repr(subp), "  "))

        return "\n".join(output)


class EvolutionaryEquilibrium(object):
    """Utils to deal with evolutionary search for equilibrium states."""

    def __init__(self, model, agentset=None,
                 popsize=6, generations=10,
                 mutation_magnitude=0.03, dropout_rate=0.5,
                 npairs=5, ngames=2,
                 log_metadata=True, log_generations=True,
                 log_payoffs=False,
                 outfolder="output", outfile_prefix="ee",
                 local_run=True,
                 stream_progress=False, progress_bar=None):

        # Saving initial params for __repr__ method
        # Excluding some params that showed explicitly in repr
        # or causes too large output
        self.__initial_params = locals()
        del(self.__initial_params["agentset"])
        del(self.__initial_params["model"])
        del(self.__initial_params["self"])

        # Boolean variable to toggle run-mode
        if stream_progress and not local_run:
            raise Exception("Can't stream progress in non-local run")
        self._local_run = local_run
        self._stream_progress = stream_progress
        self._progress_bar = progress_bar
        self._progress_bar_own = False

        # Create metadata container
        self._metadata = OrderedDict()

        self.model = copy.deepcopy(model)

        # Record general info about the evaluation
        self._metadata["model"] = self.model.__class__.__name__
        self._metadata["timestamp"] = int(time.time())

        self._generations = generations
        self._metadata["generations"] = self._generations
        self._mt_rate = mutation_magnitude
        self._metadata["mt_magnitude"] = self._mt_rate
        self._dropout_rate = dropout_rate
        self._metadata["dropout_rate"] = self._dropout_rate
        self._npairs = npairs
        self._metadata["npairs"] = self._npairs
        self._ngames = ngames
        self._metadata["ngames"] = self._ngames
        self._popsize = popsize
        self._metadata["popsize"] = self._popsize

        # Initialize current generation counter
        self._generation = 0

        # Initialize logging generations vars
        self._log_generations = log_generations
        self._log_gen_columns = []
        # Add data about logging of generations to metadata record
        self._metadata["log_generations"] = self._log_generations
        self._metadata["log_gen_columns"] = self._log_gen_columns

        # Logging payoffs
        self._log_payoffs = log_payoffs
        self._log_po_columns = []
        # Add data about logging of payoffs to metadata record
        self._metadata["log_payoffs"] = self._log_payoffs
        self._metadata["log_po_columns"] = self._log_po_columns

        # Genereate outfolder path and outname for assciated files naming
        self._outfolder = outfolder + '/' + self._metadata["model"] + '/'
        self._outname = outfile_prefix + '_' \
            + str(self._metadata["popsize"]) + '_' \
            + str(self._metadata["npairs"]) + '_' \
            + str(self._metadata["ngames"]) + '_' \
            + str(self._metadata["mt_magnitude"]).replace('.', '') + '_' \
            + str(self._metadata["dropout_rate"]).replace('.', '') + '_' \
            + str(uuid.uuid4()) + '_' \
            + str(self._metadata["timestamp"])

        if agentset is None:
            self.agentset = AgentSet(model=model)
            self.pop = Population(agentset=self.agentset,
                                  init_uniform=True,
                                  popsize=self._popsize,
                                  mutation_magnitude=self._mt_rate)
        else:
            self.agentset = copy.deepcopy(agentset)
            self.pop = Population(agentset=self.agentset,
                                  init_uniform=False,
                                  popsize=self._popsize,
                                  mutation_magnitude=self._mt_rate)

        # Initialize Generations logging
        if self._log_generations:
            self._init_log_generations()

        # Initialize Payoffs logging
        if self._log_payoffs:
            self._init_log_payoffs()

        # Record metadata
        self._record_metadata()

    def _init_progress_bar(self, total_iters):
        del(self._progress_bar)
        self._progress_bar = tqdm(total=total_iters)
        self._progress_bar_own = True

    ########################################
    # Methods dedicated to logging - START #
    ########################################

    def _init_log_payoffs(self):
        columns = []
        columns.append("Generation")
        strategies_columns = []
        payoffs_columns = []
        for pl in self.agentset:
            for idx, ds in enumerate(pl.decision_space):
                for decision in ds.decision_set:
                    str_to_append = pl.name + "-" + \
                        str(idx) + "-" + ds.type + "-" + decision
                    strategies_columns.append(str_to_append)
            str_to_append = pl.name + "-payoff"
            payoffs_columns.append(str_to_append)

        columns = columns + strategies_columns + payoffs_columns
        self._log_po_columns = columns

        self._logger = logging.getLogger("ee_po" + str(id(self)))
        self._logger.setLevel(logging.DEBUG)
        self._log_file = self._outfolder + self._outname + '_po.csv'
        fhandler = logging.FileHandler(filename=self._log_file, mode='w')
        fhandler.setLevel(logging.DEBUG)
        mhandler = MemoryHandler(capacity=2000, target=fhandler)
        self._logger_mhandler = mhandler
        self._logger.addHandler(mhandler)
        print("Loggers of EE:", self._logger.handlers)

    def _init_log_generations(self):
        """Initialize generations logging."""

        # Construct columns to gather statistics on generations
        columns = []
        columns.append("Generation")
        columns_strategies = []
        columns_fitness = []
        for idx, subp in enumerate(self.pop):
            # Record each strategy of every player in generation
            for decision in subp[0].decision_set:
                strat_str = "STRAT" + "-" + str(idx) + "-" + subp[0].pl_type \
                    + "-" + subp[0].type + "-" + decision
                columns_strategies.append(strat_str)

            # Record fitness of each strategy
            fitness_str = "FITNESS_STRAT" + "-" + \
                str(idx) + "-" + subp[0].pl_type
            columns_fitness.append(fitness_str)

        # for pl in self.agentset:
        #     for idx, ds in enumerate(pl.decision_space):
        #         for decision in ds.decision_set:
        #             str_to_append = "STRAT" + "-" + pl.name + "-" \
        #                             + str(idx) + "-" + ds.type + "-" + decision
        #             columns_strategies.append(str_to_append)
        # columns_fitness.append("Fitness")

        columns = columns + columns_strategies + columns_fitness

        self._log_gen_columns = columns
        self._metadata["log_gen_columns"] = self._log_gen_columns

        if self._local_run:
            self._init_log_gen_local()
        else:
            pass

    def _init_log_gen_local(self):
        log_gen_folder = self._outfolder + 'gen/'

        if not os.path.exists(log_gen_folder):
            os.makedirs(log_gen_folder)

        self._log_gen_file = open(log_gen_folder + self._outname
                                  + '_gen.csv', mode='w', newline='')
        self._log_gen_writer = csv.writer(self._log_gen_file)
        self._log_gen_writer.writerow(self._log_gen_columns)

    def _log_gen_write(self, population, generation):
        """Write to the log provided population
        under provided generation number."""

        rows = []
        for i in range(self._popsize):
            data = []
            data.append(generation)
            data_strategies = []
            data_fitness = []
            for subp in self.pop:

                # Record each strategy of current DS
                for strategy in subp[i]._strategy:
                    data_strategies.append(strategy)

                # Record fitness of each strategy
                data_fitness.append(subp[i].fitness)

            data = data + data_strategies + data_fitness
            rows.append(data)

        if self._local_run:
            self._log_gen_write_local(rows)
        else:
            print("Non-local run - logging generations is unsupported")

    def _log_gen_write_local(self, rows):
        self._log_gen_writer.writerows(rows)

    def _log_gen_close(self):
        if self._local_run:
            self._log_gen_file.close()
        else:
            print("Non-local run - logging generations is unsupported")

    def _record_metadata(self):
        # Create needed to write logs folders
        if not os.path.exists(self._outfolder):
            os.makedirs(self._outfolder)

        # Open file named as ee_{}_{}_{}_{}_{}.txt
        # Write to the file metadata of simulation
        with open(self._outfolder + self._outname + '_metadata.txt', mode="w") as outfile:
            json.dump(self._metadata, outfile, indent=4)

    ######################################
    # Methods dedicated to logging - END #
    ######################################

    @staticmethod
    def restore_agentset(population, agentset):
        if not isinstance(population, Population):
            raise ValueError("population must be inst of Population()")
        if not isinstance(agentset, AgentSet):
            raise ValueError("agentset must be inst of AgentSet()")

        output_players = OrderedDict()

        for pl in agentset.players:
            output_players[pl.name] = Player(pl.name, [])

        for subp in population:
            output_players[subp.pl_type].decision_space.append(subp[0])

        output = AgentSet(list(output_players.values()))
        return output

    def optimize(self, generations=None,
                 dropout_rate=None,
                 mutation_rate=None,
                 npairs=None,
                 ngames=None):

        # Substitute default params if necessary
        if generations is None:
            generations = self._generations
        if dropout_rate is None:
            dropout_rate = self._dropout_rate
        if mutation_rate is None:
            mutation_rate = self._mt_rate
        if npairs is None:
            npairs = self._npairs
        if ngames is None:
            ngames = self._ngames

        # Initialize progress bar if needed
        if self._progress_bar_own or (self._local_run and self._stream_progress
                                      and self._progress_bar is None):
            self._init_progress_bar(generations)

        for i in range(generations):
            # Calculate fitness and sort species
            self._evaluate_generation(npairs=npairs, ngames=ngames)
            # Log generation
            if self._log_generations:
                try:
                    self._log_gen_write(self.pop, self._generation)
                except ValueError:
                    logging.debug(
                        "Unable to log generation - log file is closed")

            # Create new generation
            self._update_generation_truncation(dropout_rate=dropout_rate,
                                               mutation_magnitude=mutation_rate)
            # Increment population counter
            self._generation += 1

            # Increase count in progress bar if needed
            if self._stream_progress:
                self._progress_bar.update(1)

        # Calculate weights for the final generation
        self._evaluate_generation(npairs=npairs, ngames=ngames)
        # Log final generation
        if self._log_generations:
            try:
                self._log_gen_write(self.pop, self._generation)
                self._log_gen_close()
            except ValueError:
                logging.debug("Unable to log generation - log file is closed")

        # Close progress bar if needed
        if self._progress_bar_own:
            self._progress_bar.close()

        # Put all buffered logs to their destionation
        # self._logger_mhandler.flush()
        # self.log = pd.read_csv(self._log_file, names=self.log.columns)

    def mutate_population(self, mutation_magnitude):
        # Mutate every DS in population
        for subp in self.pop:
            subp.mutate(magnitude=mutation_magnitude)
        pass

    # Run evaluations for current generation
    def _evaluate_generation(self, npairs, ngames):
        # logging.info("GENERATION %s is being processed", self._generation)

        # Update fitness values of every DS in population
        self._update_fitness(npairs=npairs, ngames=ngames)
        # logging.info("FITNESS updated")

        # Sort DSs by fitness
        for subp in self.pop:
            subp._population.sort(key=attrgetter("fitness"), reverse=True)

    # Update population for next generation
    def _update_generation_truncation(self, dropout_rate, mutation_magnitude):

        for subp in self.pop:
            # Calculate number of species to drop
            to_drop = math.ceil(len(subp) * dropout_rate)
            # Drop underperforming species
            del subp[-to_drop:]

            # Create offsprings with random sampling from best left species
            old_members = set(subp)
            for i in range(to_drop):
                offspring = copy.deepcopy(random.sample(old_members, 1)[0])
                offspring.mutate(magnitude=mutation_magnitude)
                subp.add_member(offspring)

    # Calculate fitness
    def _update_fitness(self, npairs, ngames):
        # logging.info("FITNESS UPDATE requested with %s npairs and %s ngames",
        #              npairs, ngames)

        # Make matchings and evaluate them
        matchings = self.make_matchings(npairs)

        # pdb.set_trace()
        # pool = futures.ThreadPoolExecutor(max_workers=30)
        # for matching in matchings:
        #     pool.submit(self.calculate_payoffs, matching)
        # pool.shutdown(wait=True)

        # with futures.ThreadPoolExecutor() as executor:
        #     executor.map(self.calculate_payoffs, matchings)
        #     executor.shutdown(wait=True)

        # --- One-by-one execution variant
        for matching in matchings:
            self.calculate_payoffs(matching, ngames)

        # Set evals counter and fitness for every DS in population to zero
        for subp in self.pop:
            for ds in subp:
                ds.evals = 0
                ds.fitness = 0.
                ds.fitness_averaged = False

        # Update fitness for all DSs in population
        for matching in matchings:
            for pl in matching:
                for ds in pl.decision_space:
                    ds.fitness += matching.payoffs[pl.name]
                    ds.evals += 1 * ngames

        # Averaging fitness for every DS by number of evals
        for subp in self.pop:
            for ds in subp:
                ds.fitness = ds.fitness / (ds.evals / ngames)
                ds.fitness_averaged = True

    # Algorithm to construct matchings of DSs to be evaluated
    def make_matchings(self, npairs):
        """Construct at least npairs matchings for each DSs in current population.

        OUTPUT: list of AgentSet() instances

        """
        # logging.info("Running matching algorithm for %s npairs", npairs)

        # Gather subpopulation lengths
        subp_lengths = []
        for subp in self.pop:
            subp_lengths.append(len(subp))

        logging.debug("Subp lengths are: %s", subp_lengths)

        # Calculate total number of matches to be constructed
        total_games = max(subp_lengths) * npairs
        # logging.info("Total number of matches (%s)", total_games)

        for subp in self.pop:
            # Calculate matching cap
            # (maximum number of times DS would be included in a matching)
            subp.matchings_cap = math.ceil(float(total_games) / len(subp))
            logging.debug("Matching cap for %s is %s",
                          subp, subp.matchings_cap)

            # Construct sets of DSs to be matched for each subp
            subp.ds_tobematched = set(subp)
            logging.debug("Subp's set of DSs to be matched ready, its len (%s)",
                          len(subp.ds_tobematched))

            # Set matching counters for DSs
            for ds in subp.ds_tobematched:
                ds.counter = 0

        def construct_matching_population():
            # Construct temp_pop in which selected DSs would be gathered
            temp_pop = Population(popsize=1)
            logging.debug("construct_matching_population() called")

            for subp in self.pop:
                logging.debug(
                    "SUBP (%s to choose): %s", len(
                        subp.ds_tobematched), subp)
                # Randomly choose DS
                choosen_ds = random.sample(subp.ds_tobematched, 1)[0]
                logging.debug("Chose DS: %s", choosen_ds)
                # Increment counter
                choosen_ds.counter += 1

                # Check if mathcings counter equal matchings cap
                if choosen_ds.counter >= subp.matchings_cap:
                    logging.debug("counter (%s) exceeds matching cap (%s)",
                                  choosen_ds.counter, subp.matchings_cap)
                    # If so, remove DS from set to be matched
                    subp.ds_tobematched.remove(choosen_ds)
                    logging.debug("DS excluded from DS set to be matched, its len (%s)",
                                  len(subp.ds_tobematched))

                # Add derived DS to gathering population
                temp_pop.append(SubPopulation(choosen_ds, 1))
            return temp_pop

        # Put all matchings together
        matchings = []
        for i in range(total_games):
            matching_population = construct_matching_population()
            matching_agentset = self.restore_agentset(
                matching_population, self.agentset)
            matchings.append(matching_agentset)
        logging.debug("MATCHINGS (%s) constructed", len(matchings))
        return matchings

    # Update 'payoffs' attribute of provided AgentSet
    def calculate_payoffs(self, matching, ngames):
        """Calculate payoffs for matching=AgentSet() with ngames valuations of payoffs.

        If 'ngames' > 1, payoffs averaged internally.
        """

        logging.debug("PAYOFFS CALCULATION requested")
        if not isinstance(matching, AgentSet):
            raise ValueError("matching must be inst of AgentSet()")

        payoffs_list = []
        for i in range(ngames):
            move = matching.make_move()
            payoffs_list.append(self.model.calculate_payoffs(move))

        output = {}

        # Calculate total payoffs from ngames
        for payoff in payoffs_list:
            for key, value in payoff.items():
                try:
                    output[key] += value
                except KeyError:
                    output[key] = value

        # Calculate average payoffs from total payoffs
        for key, value in output.items():
            output[key] = value / len(payoffs_list)

        matching.payoffs = output

        if self._log_payoffs:
            log = []
            log.append(self._generation)
            strategies_columns = []
            payoffs_columns = []
            for pl in matching:
                for ds in pl.decision_space:
                    strategies_columns = strategies_columns + \
                        list(ds.strategy.values())
            payoffs_columns = list(matching.payoffs.values())
            log = log + strategies_columns + payoffs_columns
            # self._logger.debug(str(','.join(map(str, log))))
            # log = pd.DataFrame(data=[log], index=None, columns=self.log.columns)
            # self.log = self.log.append(log, ignore_index=True)

    def __repr__(self):
        output = []
        # output.append("EvolutionaryEquilibrium(popsize={}, mutation_magnitude={})".format(self._popsize,
        # self._mt_rate))
        output.append(
            "EvolutionaryEquilibrium({})".format(
                self.__initial_params))
        output.append("Model: {}".format(self._metadata["model"]))
        output.append("Generation: {}".format(self._generation))
        output.append("")
        output.append("AgentSet:")
        output.append("---------")
        output.append(repr(self.agentset))
        output.append("Population:")
        output.append("---------")
        output.append(repr(self.pop))
        return "\n".join(output)

        pass
