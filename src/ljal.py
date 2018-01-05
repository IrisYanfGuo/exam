#! /usr/bin/python3

import unittest
import numpy as np
from parmap import parmap
import optimisation

from graph import *


def BoltzmannAction(evs, temp=1):
    # Prevent overflow
    # temp = max(temp, 0.3)
    cs = np.array(evs) / temp
    cs -= np.mean(cs)
    if (cs > 650).any():
        cs -= max(cs) + 650
    cs = np.cumsum(np.exp(cs))
    cs = cs / cs[-1]
    rd = np.random.uniform()
    l = len(cs)
    return min(l - np.sum(rd <= cs), l - 1)


class LJAL(object):
    def __init__(self, graph, n_actions=4, optimistic=0.0):
        self.n_actions = n_actions
        self.n_agents = len(graph.nodes)
        self.graph = graph

        self.step = 0
        ##
        self.Qs = [np.full([n_actions for i in range(0, len(n) + 1)], optimistic)
                   for n in graph.nodes]
        self.Cs = [np.zeros((len(n), n_actions), dtype=np.int)
                   for n in graph.nodes]

        ## Last values
        self.R = 0
        self.actions = np.zeros(self.n_agents)

    def alpha(self):
        return 0.1

    def reward(self, actions):
        return 1.0

    def temperature(self):
        return 1.0

    def EVs(self, agent):
        # EV = np.zeros(self.n_actions)
        Sums = np.sum(self.Cs[agent], axis=1)
        Sums[Sums == 0] = 1
        Fs = self.Cs[agent] / Sums[:, None]

        return optimisation.cython_EVs(self.n_actions, self.Qs[agent], Fs)

    def one_step(self):
        temp = self.temperature()
        self.actions = np.array([ BoltzmannAction(self.EVs(agent), temp = temp)
                                  for agent in range(0, self.n_agents) ])

        self.R = self.reward(self.actions)

        for agent in range(0, self.n_agents):
            selected_actions = [agent]
            selected_actions.extend(self.graph.successors(agent))
            selected_actions = tuple(self.actions[selected_actions])

            self.Qs[agent][selected_actions] += self.alpha() * (self.R - self.Qs[agent][selected_actions])

            for i, s in enumerate(self.graph.successors(agent)):
                self.Cs[agent][i, self.actions[s]] += 1

        self.step += 1

    def n_steps(self, n):
        result = np.empty(n)
        for step in range(0, n):
            self.one_step()
            result[step] = self.R

        return result

    def __str__(self):
        str = """
n_agents = {}
n_actions = {}
alpha = {}
step = {}
Qs = {}
Cs = {}
        """.format(self.n_agents, self.n_actions, self.alpha, self.step,
                   self.Qs, self.Cs)
        return str


def AverageR(n, getR):
    def worker(i):
        return getR()

    res = parmap(worker, range(0, n))
    return np.average(res, axis=0)
    # resR = getR()
    # for i in range(2,n+1):
    #    resR += getR()
    # return resR / n


####################
## TESTING
class TestLJALMethods(unittest.TestCase):
    def test_BoltzmannAction(self):
        self.assertTrue(BoltzmannAction([100, 0, 0]) == 0)
        self.assertTrue(BoltzmannAction([0, 100, 0, 0]) == 1)
        self.assertTrue(BoltzmannAction([0, 0, 100, 0, 0]) == 2)
        self.assertTrue(BoltzmannAction([0, 0, 0, 100]) == 3)
        half = np.mean([BoltzmannAction([10, 10]) for i in range(0, 2000)])
        self.assertTrue(0.45 < half and half < 0.55)

    def test_EVs(self):
        pass

    def test_LJAL(self):
        g = Graph(5)
        g.add_arc(1, 2)
        g.add_arc(1, 3)
        l = LJAL(g)
        self.assertTrue(np.shape(l.Qs[0]) == (4,))
        self.assertTrue(np.shape(l.Qs[1]) == (4, 4, 4))
        # print(np.shape(l.Cs[0]))
        self.assertTrue(np.shape(l.Cs[0]) == (0, 4))
        # print(np.shape(l.Cs[1]))
        self.assertTrue(np.shape(l.Cs[1]) == (2, 4))

    def test_one_step(self):
        g = Graph(5)
        g.add_arc(1, 2)
        l = LJAL(g)
        l.one_step()
        l.one_step()
        self.assertEqual(np.sum(l.Cs[1]), 2)
        # print(l)

    def test_n_steps(self):
        g = Graph(5)
        l = LJAL(g)
        R = l.n_steps(30)
        self.assertEqual(len(R), 30)
        self.assertTrue(all([x == 1.0 for x in R]))
        # print(l)

    def test_AverageR(self):
        res = AverageR(5, lambda: np.array([1, 2, 3]))
        # print(res)
        self.assertEqual(res[0], 1.0)
        self.assertEqual(res[1], 2.0)
        self.assertEqual(res[2], 3.0)


if __name__ == "__main__":
    unittest.main()
