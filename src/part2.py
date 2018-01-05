from ljal import *
import sys
from timeit import default_timer as timer
from multiprocessing import Pool
import matplotlib.pyplot as plt

class DCOPpart2(LJAL):
    def __init__(self, graph):
        super(DCOPpart2, self).__init__(graph=graph, n_actions=4, optimistic=0.0)

        self.rewards = self._set_reward(self._set_weight_mat())


    def _set_weight_mat(self):
        # here we use the structure in the paper Fig 5
        weight_mat = np.zeros([self.n_agents,self.n_agents])+0.1
        # constriants for (1,2),(1,3),(2,3), (5,6) are important
        # the matrix are symmetric, we use the half triangular
        for (i,j) in [(0,1),(0,2),(1,2),(4,5)]:
            weight_mat[i,j] = 0.9
            weight_mat[j,i] = 0.9
        for i in  range(self.n_agents):
            weight_mat[i,i] = 0

        return weight_mat



    def _set_reward(self,weight_mat):
        # the reward matrix is a 4*4*$C_7^2$ matrix
        reward_mat = [[0 for i in range(7)] for j in range(7)]
        for i in range(self.n_agents):
            for j in range(0,i):
                temp_mat = np.random.normal(0,self.n_agents*10*weight_mat[i,j],size=[self.n_actions,self.n_actions])
                reward_mat[i][j] = temp_mat
                reward_mat[j][i] = temp_mat
        return reward_mat

    def reward(self, actions):
        # the reward is the sum of the reward
        s = 0
        for i in range(self.n_agents-1):
            for j in range(i+1,self.n_agents):
                s += self.rewards[i][j][actions[i]][actions[j]]
        return s

    def temperature(self):
        return 1000* 0.96 ** self.step

    def alpha(self):
        return 0.8



# set the graph
# independent learners
n_samples = 30000
if len(sys.argv) > 1:
    n_samples = int(sys.argv[1])

steps = 200
index = [i for i in range(1, steps + 1)]

print("Running IL")
start = timer()
IL = AverageR(n_samples, lambda: DCOPpart2(graph=Graph(7)).n_steps(steps))
end = timer()
IL_delta = end - start

print("Running LJAL-1")
start = timer()
LJAL_1 = AverageR(n_samples, lambda: DCOPpart2(graph=RandomGraph(7, 2)).n_steps(steps))
end = timer()
LJAL_1_delta = end - start

print("Running LJAL-2")
LJAL2_graph = Graph(7)
LJAL2_graph.add_arc(0,1)
LJAL2_graph.add_arc(0,2)
LJAL2_graph.add_arc(1,0)
LJAL2_graph.add_arc(1,2)
LJAL2_graph.add_arc(2,0)
LJAL2_graph.add_arc(2,1)
LJAL2_graph.add_arc(4,5)
LJAL2_graph.add_arc(5,4)
start = timer()
LJAL_2 = AverageR(n_samples, lambda: DCOPpart2(graph=LJAL2_graph).n_steps(steps))
end = timer()
LJAL_2_delta = end - start

print("Running LJAL-3")
LJAL3_graph = Graph(7)
LJAL3_graph.add_arc(0,1)
LJAL3_graph.add_arc(0,2)
LJAL3_graph.add_arc(1,0)
LJAL3_graph.add_arc(1,2)
LJAL3_graph.add_arc(2,0)
LJAL3_graph.add_arc(2,1)
LJAL3_graph.add_arc(4,5)
LJAL3_graph.add_arc(5,4)
LJAL3_graph.add_arc(0,4)
LJAL3_graph.add_arc(4,0)
start = timer()
LJAL_3 = AverageR(n_samples, lambda: DCOPpart2(graph=LJAL3_graph).n_steps(steps))
end = timer()
LJAL_3_delta = end - start


print("Running JAL")
start = timer()
JAL = AverageR(n_samples, lambda: DCOPpart2(graph=FullGraph(7)).n_steps(steps))
end = timer()
JAL_delta = end - start

timing = np.array([IL_delta, LJAL_2_delta, LJAL_3_delta, JAL_delta]) / JAL_delta
print(timing)
print(timing*JAL_delta)

print("Plotting")
plt.ylim(-10, 400)
plt.plot(index, IL, index, LJAL_1, index,LJAL_2, index, LJAL_3,index,JAL)
plt.legend(['IL','LJAL_1',"LJAL_2",'LJAL_3','JAL'])
plt.ylabel('R')
plt.savefig('part2_0.96_0.8_50000run.png')
#plt.show()






