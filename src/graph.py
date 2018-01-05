#! /usr/bin/python3

import unittest
import numpy as np


class Graph(object):
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.nodes = [{} for i in range(0, n_nodes)]

        # Optimization
        self.succ_cache_valid = [ False for i in range(0,n_nodes) ]
        self.succ_cache = [ [] for i in range(0,n_nodes) ]

    def add_arc(self, src, dst, weigth = 1):
        if (dst < self.n_nodes and src != dst):
            self.nodes[src][dst] = weigth
            self.succ_cache_valid[src] = False

    def successors(self, node):
        if self.succ_cache_valid[node]:
            return self.succ_cache[node]

        self.succ_cache[node] = sorted(self.nodes[node].keys())
        self.succ_cache_valid[node] = True
        return self.succ_cache[node]


class RandomGraph(Graph):
    def __init__(self, n_nodes, n_out_edges=0):
        super(RandomGraph, self).__init__(n_nodes)
        for n in range(0, n_nodes):
            for d in np.random.choice([x for x in range(0, n_nodes) if x != n],
                                      n_out_edges, replace=False):
                self.add_arc(n, d)



class FullGraph(Graph):
    def __init__(self, n_nodes):
        super(FullGraph, self).__init__(n_nodes)
        for n1 in range(0, n_nodes):
            for n2 in range(0, n_nodes):
                if n1 != n2:
                    self.add_arc(n1, n2)


####################
## TESTING
class TestGraphMethods(unittest.TestCase):
    def test_Graph(self):
        g = Graph(10)
        self.assertTrue(isinstance(g.nodes[0], dict))

    def test_add_arc(self):
        #print("arc")
        g = Graph(10)
        g.add_arc(1, 1)
        g.add_arc(1, 2)
        g.add_arc(1, 5)
        self.assertTrue(2 in g.nodes[1])
        self.assertTrue(5 in g.nodes[1])
        self.assertFalse(1 in g.nodes[1])

    def test_add_arc1(self):
        g = Graph(10)
        g.add_arc(1, 2)
        g.add_arc(1, 5)
        self.assertEqual([2, 5], g.successors(1))

    def test_random_graph(self):
        g = RandomGraph(10, 2)
        s = sum([len(n) for n in g.nodes])
        self.assertEqual(s, 20)

    def test_full_graph(self):
        g = FullGraph(10)
        s = sum([len(n) for n in g.nodes])
        self.assertEqual(s, 90)


if __name__ == "__main__":
    unittest.main()