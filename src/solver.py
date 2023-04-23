from __future__ import annotations

import time
from dataclasses import dataclass
from math import dist
import numpy as np
import pandas as pd
import streamlit as st
from pandas import DataFrame, Series
import matplotlib.pyplot as plt


@dataclass
class TSP2D:
    n: int
    x: Series
    y: Series

    @staticmethod
    def make_random_data(n: int, xrange: int = 200, yrange: int = 200, random_seed: int = None) -> TSP2D:
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
            tour_full = [0] + tour + [0]

        x = self.x.values[tour_full]
        y = self.y.values[tour_full]

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, marker=".", markersize=15)
        if placeholder is None:
            st.pyplot(fig)
        else:
            placeholder.pyplot(fig)

    def length_of_tour(self, tour: list) -> float:
        ret = 0.0
        tour_full = [0] + tour + [0]
        for i in range(self.n):
            p1 = (self.x[tour_full[i]], self.y[tour_full[i]])
            p2 = (self.x[tour_full[i + 1]], self.y[tour_full[i + 1]])
            ret += dist(p1, p2)
        return ret
