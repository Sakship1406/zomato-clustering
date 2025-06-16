import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

st.set_page_config(
    page_title="Zomato Cuisine Recommender",
    layout="wide",
    initial_sidebar_state="auto"
)

# UI Styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    h1 {
        color: #FF4B4B;
    }
    .recommend-card {
        background-color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("zomato.csv", encoding='latin-1')
    df = df[['Restaurant Name', 'Cuisines', 'Average Cost for two', 'Aggregate rating', 'Votes']]
    df.dropna(subset=['Cuisines'], inplace=True)
    df.fillna(0, inplace=True)

    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    cuisine_features = vectorizer.fit_transform(df['Cuisines'])

    numerical = df[['Average Cost for two', 'Aggregate rating', 'Votes']]
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical)

    X = hstack([cuisine_features, numerical_scaled])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X)

    return df

df = load_data()

st.title("🍴 Zomato Cuisine Clustering Recommender")
st.markdown("##### Discover top-rated restaurants based on cuisine clusters")

cuisine_input = st.text_input("🔍 Enter a cuisine you like (e.g., Chinese, North Indian):")

if st.button("🍽 Get Recommendations"):
    matches = df[df['Cuisines'].str.contains(cuisine_input, case=False, na=False)]
    if matches.empty:
        st.error("Cuisine not found. Please try another.")
    else:
        cluster = matches['Cluster'].iloc[0]
        result = df[(df['Cluster'] == cluster) & (~df['Cuisines'].str.contains(cuisine_input))]
        top = result[['Restaurant Name', 'Cuisines', 'Aggregate rating']].sort_values(by='Aggregate rating', ascending=False).head(10)

        for index, row in top.iterrows():
            st.markdown(f"""
            <div class="recommend-card">
                <h4>🍽 {row['Restaurant Name']}</h4>
                <b>Cuisines:</b> {row['Cuisines']}<br>
                <b>Rating:</b> ⭐ {row['Aggregate rating']}
            </div>
            """, unsafe_allow_html=True)
