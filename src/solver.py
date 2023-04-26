from __future__ import annotations

from dataclasses import dataclass
from math import dist

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas import DataFrame, Series
from scipy.spatial import distance


@dataclass
class TSP2D:
    n: int
    x: Series
    y: Series
    dists: DataFrame

    @staticmethod
    def make_random_data(n: int, xrange: int = 1000, yrange: int = 1000) -> TSP2D:
        """make random 2d tsp data

        Args:
            n: number of points
            xrange: maximum x-coordinate
            yrange: maximum y-coordinate
        Returns:
            random 2D tsp data
        """
        x = pd.Series(np.random.randint(xrange, size=n))
        y = pd.Series(np.random.randint(yrange, size=n))
        coor = np.array([x, y]).T
        dists = pd.DataFrame(distance.cdist(coor, coor, metric="euclidean")).round()
        return TSP2D(n=n, x=x, y=y, dists=dists)

    @staticmethod
    def read_japan_data() -> TSP2D:
        """read japan tsp data
        Returns:
            japan tsp data
        """
        df = pd.read_csv("data/ja9847.csv")
        n = len(df)
        x = df["y"]
        y = df["x"]
        coor = np.array([x, y]).T
        dists = pd.DataFrame(distance.cdist(coor, coor, metric="euclidean")).round()
        return TSP2D(n=n, x=x, y=y, dists=dists)

    def plot_tour(self, tour: list | None = None, placeholder=None, v1=None, v4=None) -> None:
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
        ax.set_aspect("equal")
        ax.plot(x, y, linewidth=0.5)
        if v1 is not None:
            idx_1 = tour_full.index(v1)
            idx_2 = idx_1 + 1

            idx_4 = tour_full.index(v4)
            idx_3 = idx_4 + 1
            ax.plot([x[idx_1], x[idx_2]], [y[idx_1], y[idx_2]], linewidth=1.5, color="orange")
            ax.plot([x[idx_4], x[idx_3]], [y[idx_4], y[idx_3]], linewidth=1.5, color="orange")
        if placeholder is None:
            st.pyplot(fig)
        else:
            placeholder.pyplot(fig)

    def length_of_tour(self, tour: list) -> float:
        ret = 0
        tour_full = tour + [tour[0]]
        for i in range(self.n):
            p1 = (self.x[tour_full[i]], self.y[tour_full[i]])
            p2 = (self.x[tour_full[i + 1]], self.y[tour_full[i + 1]])
            ret += round(dist(p1, p2))
        return ret


def nearest_neighbor_tour(data: TSP2D) -> list:
    """return tour by nearest neighbor

    Args:
        data: condition of tsp

    Returns:
        tour by nearest neighbor
    """
    tour = [0]
    n = data.n
    dists = data.dists
    flg = [True] * n
    flg[0] = False

    for _ in range(n - 1):
        tour.append(dists.loc[tour[-1], flg].idxmin())
        flg[tour[-1]] = False
    return tour


def randomized_nearest_neighbor_tour(data: TSP2D, alpha: float = 0.0025) -> list:
    """return tour by randomized nearest neighbor

    Args:
        data: condition of tsp
        alpha: threshold

    Returns:
        tour by nearest neighbor
    """
    n = data.n
    tour = [np.random.randint(n)]
    dists = data.dists
    flg = [True] * n
    flg[tour[-1]] = False

    for _ in range(n - 1):
        points = dists.loc[0, flg].index
        dists_list = dists.loc[tour[-1], points]
        dist_max = dists_list.max()
        dist_min = dists_list.min()
        dist_threshold = dist_min + alpha * (dist_max - dist_min)
        candidates = dists_list.loc[dists_list <= dist_threshold].index
        tour.append(np.random.choice(candidates))
        flg[tour[-1]] = False
    return tour


class LocalSearch:
    data: TSP2D
    tour: list
    L: int
    dists_list: list
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

        self.dists_list = None
        self.history = []

    def build_dists(self):
        n = self.data.n
        dists = self.data.dists
        L = self.L

        self.dists_list = [dists[i].nsmallest(L + 1).iloc[1:] for i in range(n)]

    def local_search(self, init_tour: list, placeholder) -> list:
        tour = init_tour
        i = 0
        data = self.data
        n = data.n
        perm = np.random.permutation(list(range(n)))
        self.history.append(data.length_of_tour(init_tour))
        self.do_not_look = [False] * n
        while True:
            flg_improve = False
            # 始点を固定しての探索
            for _ in range(n):
                i = (i + 1) % n
                v1 = perm[i]
                if self.do_not_look[v1]:
                    continue
                flg_improve, tmp_tour, diff, v4 = self.search_2opt_single_edge(tour, v1)
                # 改善したら解を更新
                if flg_improve:
                    data.plot_tour(tour, placeholder, v1, v4)
                    tour = tmp_tour
                    self.history.append(self.history[-1] + diff)
                    break
                else:
                    self.do_not_look[v1] = True
            # すべての頂点を探索しても改善しないなら終了
            if not flg_improve:
                break

        data.plot_tour(tour, placeholder)
        return tour

    def search_2opt_single_edge(self, tour, v1) -> tuple[bool, list | None, float, int]:
        """一方の交換辺の始点を固定しての2opt近傍の探索

        Args:
            tour: 現在のツアー
            v1: 一方の交換辺の始点

        Returns:
            改善解が見つかったかどうかのフラグと、見つかった場合はその改善解と改善量
        """
        flg_improve, tmp_tour, diff, v4 = self.search_2opt_single_end(tour, v1)
        # 改善したら解を更新して次の頂点へ
        if flg_improve:
            return flg_improve, tmp_tour, diff, v4
        # 逆向きも探索
        tour_reversed = list(reversed(tour))
        flg_improve, tmp_tour, diff, v4 = self.search_2opt_single_end(tour_reversed, v1)

        return flg_improve, (tmp_tour if flg_improve else None), diff, v4

    def search_2opt_single_end(self, tour, v1) -> tuple[bool, list | None, float, int]:
        """一方の交換辺を固定しての2opt近傍の探索

        Args:
            tour: 現在のツアー
            v1: 一方の交換辺の始点

        Returns:
            改善解が見つかったかどうかのフラグと、見つかった場合はその改善解と改善量と交換した辺の情報
        """
        data = self.data
        dists = self.data.dists
        n = data.n
        idx_v1 = tour.index(v1)
        idx_v2 = (idx_v1 + 1) % n
        v2 = tour[idx_v2]
        d12 = dists.loc[v1, v2]
        for v4, d14 in self.dists_list[v1][self.dists_list[v1] < d12].items():
            # d14がd12以上になったら、それ以降は探索しなくていい
            if d14 >= d12:
                break
            idx_v4 = tour.index(v4)
            idx_v3 = (idx_v4 + 1) % n
            v3 = tour[idx_v3]
            if len({v1, v2, v3, v4}) != 4:
                continue
            d34 = dists.loc[v3, v4]
            d23 = dists.loc[v2, v3]

            if d23 + d14 < d12 + d34:
                self.do_not_look[v1] = False
                self.do_not_look[v2] = False
                self.do_not_look[v3] = False
                self.do_not_look[v4] = False
                return True, move_to_2opt_neighbor(tour, idx_v1, idx_v2, idx_v3, idx_v4), d23 + d14 - d12 - d34, v4

        return False, None, 0.0, 0


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
