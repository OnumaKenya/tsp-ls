from src.solver import TSP2D
import numpy as np
import pandas as pd
from itertools import product


# tsp_data = TSP2D.read_japan_data()
tsp_data = TSP2D.make_random_data(2000)
n = tsp_data.n
dists = tsp_data.dists

mincost = pd.Series([dists[i][1] for i in range(1, n)], index=range(1, n))
prec = pd.Series(1, index=range(1, n))
used = pd.Series(False, index=range(1, n))

used[1] = True
tmp_dists = dists.iloc[1:, 1:]

for _ in range(n - 2):
    tmp_min = np.inf
    v = mincost[~used].idxmin()
    used[v] = True
    # mincostを更新
    to_update = ~used & (tmp_dists[v] < mincost)
    mincost[to_update] = tmp_dists[v][to_update]
    prec[to_update] = v

beta = pd.DataFrame(0, index=range(n), columns=range(n))

def calc_beta(i, j):
    if beta[i][j] >= 0:
        return beta[i][j]
    if i > j:
        return calc_beta(j, i)
    if i == prec[j] or j == prec[i]:
        beta[i][j] = dists[i][j]
        beta[j][i] = beta[i][j]
        return beta[i][j]
    beta[i][j] = max(calc_beta(i, prec[j]), dists[j][prec[j]])
    beta[j][i] = beta[i][j]
    return beta[i][j]


for i, j in product(range(1, n), repeat=2):
    calc_beta(i, j)

alpha = dists - beta
best2 = dists.loc[0, 1:].nsmallest(2)
for i in range(n):
    alpha[i][i] = 0
    alpha[0][i] = max(0, dists[0][i] - best2.iloc[1])
    alpha[i][0] = alpha[0][i]
print(alpha)
