import streamlit as st
from clustering import load_and_clean_data, cluster_cuisines, visualize

st.title("🍽 Zomato Cuisine Clustering")

df = load_and_clean_data("zomato.csv")
st.write("### Sample Data", df.head())

n = st.slider("Choose number of clusters", 2, 10, 5)
clustered_df, model = cluster_cuisines(df, n)
st.write("### Clustered Data", clustered_df.head())

if st.button("Show Cluster Plot"):
    visualize(clustered_df)

