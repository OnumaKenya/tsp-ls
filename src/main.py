import streamlit as st
from solver import TSP2D, LocalSearch, kick_double_bridge, nearest_neighbor_tour
import matplotlib.pyplot as plt
st.header("Solve TSP by ILS")

n = st.number_input("number of points", min_value=20, max_value=3000, value=100, step=1)
ls_iterations = st.number_input("number of iterations", min_value=1, max_value=50, value=10, step=1)
random_seed = st.number_input("random seed", min_value=1, value=42, step=1)
L = st.number_input("size of distance list", min_value=10, max_value=1000, value=20, step=1)

if st.button("Go!"):
    tsp_data = TSP2D.make_random_data(n, random_seed=random_seed)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Opt Tour")
        placeholder_opt = st.empty()
        placeholder_opt_obj = st.empty()
    with col2:
        st.subheader("Current Local Search")
        placeholder = st.empty()
        placeholder_obj = st.empty()
    st.subheader("length of tour")
    obj_plot = st.empty()
    obj_plot2 = st.empty()

    ls = LocalSearch(tsp_data, L=L)
    ls.build_dists()

    opt_tour = nearest_neighbor_tour(tsp_data)
    tsp_data.plot_tour(opt_tour, placeholder_opt)
    placeholder_opt_obj.info(f"tour length: {tsp_data.length_of_tour(opt_tour): .3f} (initial)")

    init_tour = opt_tour
    itr_num = []
    tour_obj = []
    opt_len = tsp_data.length_of_tour(init_tour)

    for i in range(1, ls_iterations + 1):
        tour = ls.local_search(init_tour, placeholder)
        placeholder_obj.info(f"tour length: {tsp_data.length_of_tour(tour): .3f} (iteration No.{i})")
        tour_len = tsp_data.length_of_tour(tour)

        itr_num.append(i)
        tour_obj.append(tour_len)
        if tour_len < opt_len:
            opt_tour = tour
            opt_len = tour_len
            tsp_data.plot_tour(opt_tour, placeholder_opt)
            placeholder_opt_obj.info(f"tour length: {tour_len :.3f} (iteration No.{i})")

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

        init_tour = kick_double_bridge(tour)
    st.success("Optimization Done!")
