from __future__ import annotations

import bisect
from dataclasses import dataclass
from math import dist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas import Series


@dataclass
class TSP2D:
    n: int
    x: Series
    y: Series

    @staticmethod
    def make_random_data(n: int, xrange: int = 1000, yrange: int = 1000, random_seed: int = None) -> TSP2D:
        """make random 2d tsp data

        Args:
            n: number of points
            xrange: maximum x-coordinate
            yrange: maximum y-coordinate
            random_seed: random seed
        Returns:
            random 2D tsp data
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        x = pd.Series(np.random.randint(xrange, size=n))
        y = pd.Series(np.random.randint(yrange, size=n))
        return TSP2D(n=n, x=x, y=y)

    def plot_tour(self, tour: list | None = None, placeholder=None) -> None:
        """plot tour

        Args:
            tour: list which shows tour
            placeholder: placeholder of plot

        """
        if tour is None:
            tour_full = list(range(self.n)) + [0]
        else:
            if len(tour) != self.n:
                raise RuntimeError
            tour_full = tour + [tour[0]]

        x = self.x.values[tour_full]
        y = self.y.values[tour_full]

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, marker=".")
        if placeholder is None:
            st.pyplot(fig)
        else:
            placeholder.pyplot(fig)

    def length_of_tour(self, tour: list) -> float:
        ret = 0.0
        tour_full = tour + [tour[0]]
        for i in range(self.n):
            p1 = (self.x[tour_full[i]], self.y[tour_full[i]])
            p2 = (self.x[tour_full[i + 1]], self.y[tour_full[i + 1]])
            ret += dist(p1, p2)
        return ret


def nearest_neighbor_tour(data: TSP2D) -> list:
    """return tour by nearest neighbor

    Args:
        data: condition of tsp

    Returns:
        tour by nearest neighbor
    """
    tour = [0]
    appended = set(tour)
    x = data.x
    y = data.y
    n = data.n
    for _ in range(n - 1):
        p1 = (x[tour[-1]], y[tour[-1]])
        p2_dist = np.inf
        p2 = -1
        for i in range(n):
            if i in appended:
                continue
            tmp = (x[i], y[i])
            if dist(p1, tmp) < p2_dist:
                p2 = i
                p2_dist = dist(p1, tmp)

        tour.append(p2)
        appended.add(p2)
    return tour


class LocalSearch:
    data: TSP2D
    tour: list
    L: int
    dists: list
    do_not_look: list
    history: list

    def __init__(self, data: TSP2D, L: int | None = None):
        self.data = data
        n = data.n
        self.tour = list(range(n))
        if L is None:
            self.L = n
        else:
            self.L = L

        self.dists = [[] for _ in range(n)]
        self.history = []

    def build_dists(self):
        n = self.data.n
        x = self.data.x
        y = self.data.y
        L = self.L

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = dist((x[i], y[i]), (x[j], y[j]))
                bisect.insort(self.dists[i], (d, j))
                if len(self.dists[i]) > L:
                    self.dists[i] = self.dists[i][:L]

    def local_search(self, init_tour: list, placeholder) -> list:
        tour = init_tour
        v1 = -1
        data = self.data
        n = data.n
        self.history.append(data.length_of_tour(init_tour))
        self.do_not_look = [False] * n
        while True:
            flg_improve = False
            # 始点を固定しての探索
            for _ in range(n):
                v1 = (v1 + 1) % n
                if self.do_not_look[v1]:
                    continue
                flg_improve, tmp_tour, diff = self.search_2opt_single_edge(tour, v1)
                # 改善したら解を更新
                if flg_improve:
                    tour = tmp_tour
                    # TODO: 効率化
                    self.history.append(self.history[-1] + diff)
                    break
                else:
                    self.do_not_look[v1] = True
            # すべての頂点を探索しても改善しないなら終了
            if not flg_improve:
                break

            data.plot_tour(tour, placeholder)
        return tour

    def search_2opt_single_edge(self, tour, v1) -> tuple[bool, list | None, float]:
        """一方の交換辺の始点を固定しての2opt近傍の探索

        Args:
            tour: 現在のツアー
            v1: 一方の交換辺の始点

        Returns:
            改善解が見つかったかどうかのフラグと、見つかった場合はその改善解と改善量
        """
        flg_improve, tmp_tour, diff = self.search_2opt_single_end(tour, v1)
        # 改善したら解を更新して次の頂点へ
        if flg_improve:
            return flg_improve, tmp_tour, diff
        # 逆向きも探索
        tour_reversed = list(reversed(tour))
        flg_improve, tmp_tour, diff = self.search_2opt_single_end(tour_reversed, v1)

        return flg_improve, (tmp_tour if flg_improve else None), diff

    def search_2opt_single_end(self, tour, v1) -> tuple[bool, list | None, float]:
        """一方の交換辺を固定しての2opt近傍の探索

        Args:
            tour: 現在のツアー
            v1: 一方の交換辺の始点

        Returns:
            改善解が見つかったかどうかのフラグと、見つかった場合はその改善解と改善量
        """
        data = self.data
        n = data.n
        x = data.x
        y = data.y
        idx_v1 = tour.index(v1)
        idx_v2 = (idx_v1 + 1) % n
        v2 = tour[idx_v2]
        d12 = dist((x[v1], y[v1]), (x[v2], y[v2]))
        for d14, v4 in self.dists[v1]:
            # d14がd12以上になったら、それ以降は探索しなくていい
            if d14 >= d12:
                break
            idx_v4 = tour.index(v4)
            idx_v3 = (idx_v4 + 1) % n
            v3 = tour[idx_v3]
            if len({v1, v2, v3, v4}) != 4:
                continue
            d34 = dist((x[v3], y[v3]), (x[v4], y[v4]))
            d23 = dist((x[v2], y[v2]), (x[v3], y[v3]))

            if d23 + d14 < d12 + d34:
                self.do_not_look[v1] = False
                self.do_not_look[v2] = False
                self.do_not_look[v3] = False
                self.do_not_look[v4] = False
                return True, move_to_2opt_neighbor(tour, idx_v1, idx_v2, idx_v3, idx_v4), d23 + d14 - d12 - d34

        return False, None, 0.0


def move_to_2opt_neighbor(tour: list, i1: int, i2: int, i3: int, i4: int) -> list:
    """指定した2opt近傍へ移動する"""
    if i2 < i4:
        return tour[:i2] + list(reversed(tour[i2 : i4 + 1])) + tour[i4 + 1 :]
    else:
        return tour[:i3] + list(reversed(tour[i3 : i1 + 1])) + tour[i1 + 1 :]


def kick_double_bridge(tour: list) -> list:
    """double bridgeにより解を変形させる"""
    n = len(tour)
    if n < 8:
        raise RuntimeError("Few edges to kick double bridge.")

    kick_edges = []
    unused_set = set(range(n))
    while len(kick_edges) < 4:
        ln = len(unused_set)
        i = np.random.randint(ln)
        u = sorted(unused_set)[i]
        unused_set = unused_set - {u, (u + 1) % n, (u + n - 1) % n}
        kick_edges.append((u, (u + 1) % n))

    kick_edges.sort()
    ret = []
    for i in range(2, -1, -1):
        for j in range(kick_edges[i][1], kick_edges[i + 1][0] + 1):
            ret.append(tour[j])

    if kick_edges[3][1] == 0:
        ret += tour[: kick_edges[0][0] + 1]
    else:
        ret += tour[kick_edges[3][1] :] + tour[: kick_edges[0][0] + 1]
    return ret
