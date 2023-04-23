import pandas as pd
import streamlit as st
from itertools import permutations
from solver import TSP2D
import numpy as np
import matplotlib.pyplot as plt

n = 10
tsp_data = TSP2D.make_random_data(n)
opt_len = np.inf
opt_tour = None

placeholder = st.empty()
placeholder_tour = st.empty()
placeholder_obj = st.empty()

for tour in permutations(range(1, n)):
    if tsp_data.length_of_tour(list(tour)) < opt_len:
        opt_len = tsp_data.length_of_tour(list(tour))
        opt_tour = tour
        tsp_data.plot_tour(list(opt_tour), placeholder=placeholder)
        placeholder_tour.info(f"current tour:{(0,) + opt_tour}")
        placeholder_obj.info(f"current objective: {opt_len}")

st.success("Optimization Done!")

