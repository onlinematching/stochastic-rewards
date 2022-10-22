import collections
import sympy
import functools
import sys
from copy import deepcopy
import itertools
import math
sys.setrecursionlimit(10000)

# u = {0, 1, 2, ..., u_count - 1}
# v = {0, 1, 2, ..., v_count - 1}
# and edges = [ (ui, vj), ... ]
Edges = list[tuple[int, int]]
Nodes = list[int]


class Graph:
    def __init__(self, edges: Edges):
        self.U: Nodes = []
        self.V: Nodes = []
        self.edges: Edges = []
        for u, v in edges:
            assert (u, v) not in self.edges
            self.edges.append((u, v))
            if u not in self.U:
                self.U.append(u)
            if v not in self.V:
                self.V.append(v)

    def opt(self) -> int:
        ...

    def expected_success(self):
        if len(self.V) == 0:
            assert len(self.U) == 0
            return 0
        v0 = self.V[0]
        v0_adj = []
        edges_expect_v0 = []
        for e_u, e_v in self.edges:
            if e_v == v0:
                v0_adj.append(e_u)
            else:
                edges_expect_v0.append((e_u, e_v))
        exps_dis = Graph._expected_success_distribution(len(v0_adj))
        exps = 0
        for k in range(len(v0_adj) + 1):
            exps += exps_dis[k] * k
            pk = exps_dis[k] / math.comb(len(self.U), k)
            for it in itertools.combinations(v0_adj, k):
                edges_expect_v0_and_adj = []
                for e_u, e_v in edges_expect_v0:
                    if e_u not in it:
                        edges_expect_v0_and_adj.append((e_u, e_v))
                next_graph = Graph(edges_expect_v0_and_adj)
                exps += pk * next_graph.expected_success()
        return sympy.simplify(exps)

    @functools.cache
    def _expected_success_distribution(N):
        distr = []
        p = 0
        for k in range(N):
            pk = sympy.exp(-1) / math.factorial(k)
            distr.append(pk)
            p += pk
        distr.append(1 - p)
        return distr


def G(n: int) -> Graph:
    return Graph([(u, v) for v in range(n) for u in range(v, n)])


def main():
    g = G(3)
    print(g.expected_success())


if __name__ == "__main__":
    main()
