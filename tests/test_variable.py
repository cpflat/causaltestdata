#!/usr/bin/env python
# coding: utf-8

import unittest

from causaltestdata import variable
import networkx as nx


class TestVariable(unittest.TestCase):

    def test_value(self):
        g = nx.DiGraph()
        g.add_nodes_from([1, 2, 3, 4, 5])
        g.add_edge(1, 2, weight=0.3)
        g.add_edge(2, 3, weight=0.5)
        g.add_edge(4, 3, weight=0.1)
        g.add_edge(5, 4, weight=0.1)
        defaults = {}

        df = variable.generate_all(g, defaults)
