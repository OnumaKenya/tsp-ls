import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from solver import TSP2D, LocalSearch, kick_double_bridge, nearest_neighbor_tour, randomized_nearest_neighbor_tour

st.header("Solve TSP by LS")

data_flg = st.selectbox("select data type", ("Japan Data", "Make Random"))
if data_flg == "Make Random":
    n = st.number_input("number of points", min_value=20, max_value=3000, value=100, step=1)
method = st.selectbox("select method", ("ILS", "GRASP"))
ls_iterations = st.number_input("number of iterations", min_value=1, max_value=50, value=30, step=1)
random_seed = st.number_input("random seed", min_value=0, value=42, step=1)

if st.button("Go!"):
    np.random.seed(random_seed)
    if data_flg == "Make Random":
        tsp_data = TSP2D.make_random_data(n)
    else:
        tsp_data = TSP2D.read_japan_data()

    col1, col2 = st.columns(2)
    st.subheader("Current Opt Tour")
    placeholder_opt = st.empty()
    placeholder_opt_obj = st.empty()
    st.subheader("Current Local Search")
    placeholder = st.empty()
    placeholder_obj = st.empty()
    st.subheader("length of tour")
    obj_plot = st.empty()
    obj_plot2 = st.empty()
    if method == "ILS":
        opt_tour = nearest_neighbor_tour(tsp_data)
    else:
        opt_tour = randomized_nearest_neighbor_tour(tsp_data)
    tsp_data.plot_tour(opt_tour, placeholder_opt)
    placeholder_opt_obj.info(f"tour length: {tsp_data.length_of_tour(opt_tour)} (initial)")

    ls = LocalSearch(tsp_data)
    ls.build_dists()

    init_tour = opt_tour
    itr_num = []
    tour_obj = []
    opt_len = tsp_data.length_of_tour(init_tour)

    for i in range(1, ls_iterations + 1):
        tour = ls.local_search(init_tour, placeholder)
        placeholder_obj.info(f"tour length: {tsp_data.length_of_tour(tour)} (iteration No.{i})")
        tour_len = tsp_data.length_of_tour(tour)

        itr_num.append(i)
        tour_obj.append(tour_len)
        if tour_len < opt_len:
            opt_tour = tour
            opt_len = tour_len
            tsp_data.plot_tour(opt_tour, placeholder_opt)
            placeholder_opt_obj.info(f"tour length: {tour_len} (iteration No.{i})")

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.set_xlabel("Number of local search")
        ax.set_ylabel("length of tour")
        ax.plot(itr_num, tour_obj, marker=".")
        obj_plot.pyplot(fig)

        plt.close()
        fig = plt.figure()
        ax = fig.add_subplot(3, 1, 1)
        ax.set_xlabel("Number of move to neighbor")
        ax.set_ylabel("length of tour")
        ax.plot(list(range(len(ls.history))), ls.history)
        obj_plot2.pyplot(fig)

        if method == "ILS":
            init_tour = kick_double_bridge(tour)
        else:
            init_tour = randomized_nearest_neighbor_tour(tsp_data)
    st.success("Optimization Done!")
