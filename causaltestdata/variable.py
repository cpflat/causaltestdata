import datetime
import numpy as np

import pandas as pd
import networkx as nx
from . import dtutil

TYPE_VARIABLE = "variable"
TYPE_BINARY_VARIABLE = "binary"
TYPE_COUNTABLE_VARIABLE = "countable"
TYPE_TIMESERIES_EVENT = "tsevent"


class Variable(object):

    def __init__(self, node_id, node_data, index, defaults, observable=True):
        self._node_id = node_id
        self._node_data = node_data
        self._index = index
        self._defaults = defaults
        self._observable = observable

        self._size = len(self._index)
        self._values = None

    @property
    def values(self):
        return self._values

    def get_df(self):
        return pd.DataFrame(self._values, index=self._index)

    def _get_spec(self, key, default_value):
        if key in self._node_data:
            return self._node_data[key]
        elif key in self._defaults:
            return self._defaults[key]
        else:
            return default_value

    def _rand(self):
        # TODO custoized random distribution
        return self._rand_default()

    def _rand_default(self):
        return self._rand_gauss()

    def _rand_gauss(self):
        scale = self._get_spec("gaussian_scale", 1)
        loc = self._get_spec("gaussian_loc", 0)
        return np.random.normal(loc, scale, self._size)

    def _rand_poisson(self):
        lam = self._get_spec("poisson_lambd", 10)
        return np.random.poisson(lam, self._size)

    def generate(self, effects):
        val = self._rand()
        for edge_data, variable in effects:
            assert variable.values is not None
            weight = edge_data["weight"]
            val += weight * variable.values
        self._values = val
        return val


class BinaryVariable(Variable):

    def _rand(self):
        prob = self._get_spec("binary_prob", 0.5)
        return np.random.rand() < prob

    def _binary_error(self):
        msg = "more than 1.0 total effect on binary variable {0}".format(
            self._node_id)
        raise ValueError(msg)

    def generate(self, effects):
        # On binary variable, total effect + random factor must be always 1.0
        sum_effect_weight = sum(edge["weight"] for edge, _ in effects)
        if sum_effect_weight > 1:
            self._binary_error()
        rand_weight = 1 - sum_effect_weight

        # calculate probability vector
        a_prob = rand_weight * self._rand()
        for edge_data, variable in effects:
            assert variable.values is not None
            weight = edge_data["weight"]
            a_prob += weight * variable.values

        # 1 if rand is smaller than probability vector (0 < rand <= 1)
        # else 0
        self._values = (np.random.rand(self._size) < a_prob).astype(int)
        return self._values


class CountableVariable(Variable):

    def _rand_default(self):
        # poisson function in default
        return self._rand_poisson()

    def generate(self, effects):
        val = self._rand()
        for edge_data, variable in effects:
            assert variable.values is not None
            weight = edge_data["weight"]
            # e.g. effect of weight 0.5 from node of value 3
            # means 3 trials for 50% chance of increasing 1
            # -> binomial distribution
            val += np.random.binomial(variable.values, weight)
        self._values = val
        return val


class TimeSeriesEventVariable(Variable):

    def __init__(self, node_id, node_data, index, defaults, observable=True):
        super().__init__(node_id, node_data, index, defaults, observable)
        assert "dt_range" in defaults
        self._dt_range = defaults["dt_range"]
        self._dt_interval = defaults["dt_interval"]
        self._ts = None

    @property
    def ts(self):
        return self._ts

    @staticmethod
    def _rand_next_exp(dt, lambd):
        # lambd (float): The average of appearance of generated event per a day.

        #expv = random.expovariate(lambd)
        expv = np.random.exponential(1 / lambd)
        return dt + datetime.timedelta(seconds=1) * int(
            24 * 60 * 60 * expv)

    def _rand_exp_interval(self):
        lambd = self._get_spec("tsevent_lambd", 10)

        tmp_dt = self._dt_range[0]
        tmp_dt = self._rand_next_exp(tmp_dt, lambd)
        while tmp_dt < self._dt_range[1]:
            yield tmp_dt
            tmp_dt = self._rand_next_exp(tmp_dt, lambd)

    def _rand_ts(self):
        # TODO custoized random distribution
        return list(self._rand_exp_interval())

    def _delay(self, dt):
        # TODO randomized delay
        delay = self._get_spec("delay", datetime.timedelta(0))
        return dt + delay

    def generate(self, effects):
        tmp_ts = self._rand_ts()
        for edge_data, variable in effects:
            assert variable.values is not None
            weight = edge_data["weight"]
            if "delay" in edge_data:
                delay = edge_data["delay"]
            else:
                delay = self._defaults["delay"]

            for dt in variable.ts:
                if np.random.rand() < weight:
                    tmp_ts.append(dt + delay)

        self._ts = sorted(tmp_ts)
        self._values = dtutil.discretize_sequential(self._ts, self._dt_range,
                                                    self._dt_interval)
        return self._values


def init_variable(node_id, node_data, variable_index, defaults):
    if "type" in node_data:
        nodetype = node_data["type"]
    else:
        nodetype = defaults["default_type"]
    observable = "hidden" not in node_data or node_data["hidden"] is False

    if nodetype == TYPE_VARIABLE:
        return Variable(node_id, node_data, variable_index, defaults,
                        observable=observable)
    if nodetype == TYPE_BINARY_VARIABLE:
        return BinaryVariable(node_id, node_data, variable_index, defaults,
                              observable=observable)
    elif nodetype == TYPE_COUNTABLE_VARIABLE:
        return CountableVariable(node_id, node_data, variable_index, defaults,
                                 observable=observable)
    elif nodetype == TYPE_TIMESERIES_EVENT:
        return TimeSeriesEventVariable(node_id, node_data, variable_index,
                                       defaults, observable=observable)
    else:
        raise ValueError


def generate_all(g, defaults):
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("input graph must be a DAG")

    from .defaults import default_setup
    defaults = default_setup(g, defaults)

    variable_index = defaults["variable_index"]

    variables = {}
    for node_id, node_data in g.nodes(data=True):
        variables[node_id] = init_variable(node_id, node_data,
                                           variable_index, defaults)

    # determine values by causal order (parents -> childs)
    for node_id in nx.topological_sort(g):
        effects = []
        parent_ids = list(g.predecessors(node_id))
        for parent_node_id in parent_ids:
            edge_data = g.get_edge_data(parent_node_id, node_id)
            effects.append([edge_data, variables[parent_node_id]])

        # print("determine {0} with parents {1}".format(node_id, parent_ids))
        variables[node_id].generate(effects)

    # generate dataframe
    l_tmp_df = []
    for node_id, variable in variables.items():
        tmp_df = variable.get_df()
        tmp_df.columns = [node_id, ]
        l_tmp_df.append(tmp_df)
    return pd.concat(l_tmp_df, axis=1)
