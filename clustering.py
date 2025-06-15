import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_and_clean_data(path):
    df = pd.read_csv(path, encoding='latin1')
    df = df[['Restaurant Name', 'Cuisines', 'Aggregate rating']]
    df = df.dropna()
    df = df[df['Cuisines'] != '']
    return df

def cluster_cuisines(df, n_clusters=5):
    tfidf = TfidfVectorizer(stop_words='english')
    cuisine_matrix = tfidf.fit_transform(df['Cuisines'])

    features = pd.DataFrame(cuisine_matrix.toarray())
    features['Rating'] = df['Aggregate rating'].values

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(features)
    return df, kmeans

def visualize(df):
    tfidf = TfidfVectorizer(stop_words='english')
    cuisine_matrix = tfidf.fit_transform(df['Cuisines'])

    features = pd.DataFrame(cuisine_matrix.toarray())
    features['Rating'] = df['Aggregate rating'].values

    reduced = PCA(n_components=2).fit_transform(features)
    plt.scatter(reduced[:, 0], reduced[:, 1], c=df['Cluster'], cmap='rainbow')
    plt.title("Cuisine Clusters")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.show()
