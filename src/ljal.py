#! /usr/bin/python3

import unittest
import numpy as np

from graph import *

def BoltzmannAction(evs, temp=1):
    cs = np.cumsum(np.exp(np.array(evs)/temp))
    rd = np.random.uniform(0,cs[-1])
    return len(cs) - np.sum(rd <= cs)

def EVs(Q, N):
    ## np.average(Q, axis=1, weigths=N) but handling the case where
    ## all N are zeroes
    sum_n = np.sum(N,axis=1)
    sum_n[sum_n == 0] = 1
    return np.sum(Q, axis=1) / sum_n
        

class LJAL(object):

    def __init__(self, graph, n_actions = 4, alpha = 0.1):
        self.n_actions = n_actions
        self.alpha = alpha
        self.n_agents = len(graph.nodes)
        self.graph = graph
        self.step = 0
        ## 2D matrix action x action^#successors(agent)
        self.Qs = [ np.zeros((n_actions, n_actions**len(n)))
                    for n in graph.nodes ]
        self.Ns = [ np.zeros((n_actions, n_actions**len(n)), dtype=np.int)
                    for n in graph.nodes ]

    def reward(self, actions):
        return 1

    def _y(self, agent, actions):
        """
        Compute the y axis of Q and N from the other agents actions
        """
        sel_actions = actions[self.graph.successors(agent)]
        exponents = np.full(len(sel_actions), self.n_actions)
        if len(exponents > 0):
            exponents[0] = 1
        exponents = np.cumprod(exponents)
        return np.sum(sel_actions * exponents)
        
    def one_step(self):
        actions = np.array([ BoltzmannAction(EVs(self.Qs[agent], self.Ns[agent]))
                             for agent in range(0, self.n_agents) ])

        self.R = self.reward(actions)

        for agent in range(0, self.n_agents):
            x = actions[agent]
            y = self._y(agent, actions)
            self.Qs[agent][x, y] += self.alpha * (self.R - self.Qs[agent][x, y])
            self.Ns[agent][x, y] += 1
        
        self.step += 1

    def __str__(self):
        str = """
n_agents = {}
n_actions = {}
alpha = {}
step = {}
Qs = {}
Ns = {}
        """.format(self.n_agents, self.n_actions, self.alpha, self.step,
                   self.Qs, self.Ns)
        return str

####################
## TESTING
class TestLJALMethods(unittest.TestCase):

    def test_BoltzmannAction(self):
        self.assertTrue(BoltzmannAction([100,0,0]) == 0)
        self.assertTrue(BoltzmannAction([0,100,0,0]) == 1)
        self.assertTrue(BoltzmannAction([0,0,100,0,0]) == 2)
        self.assertTrue(BoltzmannAction([0,0,0,100]) == 3)
        half = np.mean([BoltzmannAction([10,10]) for i in range(0,2000)])
        self.assertTrue(0.45 < half and half < 0.55)

    def test_EVs(self):
        Q = np.array([[ 1.,  2.], [ 0.,  0.]])
        N = np.array([[1, 1], [0, 0]])
        self.assertTrue( all( EVs(Q, N) == [1.5, 0] ) )

        
    def test_LJAL(self):
        g = Graph(5)
        g.add_arc(1,2)
        g.add_arc(1,3)
        l = LJAL(g)              
        self.assertTrue(np.shape(l.Qs[0]) == (4,1))
        self.assertTrue(np.shape(l.Qs[1]) == (4,4**2))
        self.assertTrue(np.shape(l.Ns[0]) == (4,1))
        self.assertTrue(np.shape(l.Ns[1]) == (4,4**2))

    def test_one_step(self):
        g = Graph(5)
        l = LJAL(g)
        l.one_step()
        ## print(l)
        l.one_step()
        ##print(l)
        

if __name__ == "__main__":
    unittest.main()
  