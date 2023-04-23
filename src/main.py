import streamlit as st
from solver import TSP2D, LocalSearch, kick_double_bridge, nearest_neighbor_tour

st.header("Solve TSP by ILS")

n = st.number_input("number of points", min_value=20, max_value=1000, value=100, step=1)
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

    ls = LocalSearch(tsp_data, L=L)
    ls.build_dists()
    opt_tour = nearest_neighbor_tour(tsp_data)
    tsp_data.plot_tour(opt_tour, placeholder_opt)
    placeholder_opt_obj.info(f"tour length: {tsp_data.length_of_tour(opt_tour): .3f} (initial)")

    init_tour = opt_tour

    def ordinal(num):
        return "%d%s" % (num, "tsnrhtdd"[(num / 10 % 10 != 1) * (num % 10 < 4) * num % 10 :: 4])

    for i in range(1, ls_iterations + 1):
        tour = ls.local_search(init_tour, placeholder)
        placeholder_obj.info(f"tour length: {tsp_data.length_of_tour(tour): .3f} ({ordinal(i)} iteration)")
        if tsp_data.length_of_tour(tour) < tsp_data.length_of_tour(opt_tour):
            opt_tour = tour
            tsp_data.plot_tour(opt_tour, placeholder_opt)
            placeholder_opt_obj.info(f"tour length: {tsp_data.length_of_tour(opt_tour) :.3f} ({ordinal(i)} iteration)")
        init_tour = kick_double_bridge(tour)
    st.success("Optimization Done!")
