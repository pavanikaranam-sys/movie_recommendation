import streamlit as st
import pandas as pd
import pickle
import numpy as np
import requests
from sklearn.neighbors import NearestNeighbors

def get_movie_details(movie_title, fallback_rating):
    
    api_key = "ba55cc1da533c43bfeef23f5c4f0095e" 
    base_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    
    poster = "https://via.placeholder.com/500x750?text=No+Poster"
    rating = fallback_rating 
    
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            if data['results']:
                result = data['results'][0]
                if result.get('poster_path'):
                    poster = f"https://image.tmdb.org/t/p/w500{result['poster_path']}"
                
                api_vote = result.get('vote_average', 0)
                if api_vote > 0:
                    rating = api_vote
    except:
        pass
    return poster, rating

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #e50914;
        color: white;
    }
    .stSelectbox div[data-baseweb="select"] {
        cursor: pointer;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    nn_model = pickle.load(open("nn_model.pkl", "rb"))
    final_matrix = pickle.load(open("final_matrix.pkl", "rb"))
    data = pd.read_csv("movies_cleaned.csv")
    return nn_model, final_matrix, data

nn_model, final_matrix, data = load_model()

def recommend(movie_title, top_n=10, language=None):
    matches = data[data['title'].str.lower() == movie_title.lower()]
    if len(matches) == 0:
        return None

    idx = matches.index[0]
    distances, indices = nn_model.kneighbors(final_matrix[idx], n_neighbors=300)
    rec_indices = indices.flatten()[1:]
    rec_distances = distances.flatten()[1:]

    recommendations = data.iloc[rec_indices].copy()
    recommendations['similarity'] = 1 - rec_distances
    recommendations['final_score'] = (0.7 * recommendations['similarity'] + 0.3 * recommendations['weighted_score'])
    recommendations = recommendations.sort_values(by='final_score', ascending=False)

    if language:
        lang_movies = recommendations[recommendations['original_language'] == language]
        other_movies = recommendations[recommendations['original_language'] != language]
        recommendations = pd.concat([lang_movies, other_movies])

    return recommendations.head(top_n)

st.title("🎬 Movie Recommendation System")

movie_input = st.selectbox("Search or Select a Movie", data['title'].values)

with st.sidebar:
    st.header("Settings")
    language_input = st.text_input("Language (e.g., en, hi, fr)", value="")
    top_n = st.slider("Number of Recommendations", 5, 50, 10)

if st.button("Recommend"):
    results = recommend(movie_input, top_n, language_input)

    if results is not None:
        st.success(f"Top {top_n} movies for you:")
        
        for i in range(0, len(results), 5):
            cols = st.columns(5)
            chunk = results.iloc[i : i + 5]
            
            for j, (index, row) in enumerate(chunk.iterrows()):
                with cols[j]:
                    csv_rating = round(row.get('weighted_score', 0) * 10, 1)
                    poster_url, final_rating = get_movie_details(row['title'], csv_rating)
                    
                    st.image(poster_url, use_container_width=True)

                    display_title = row['title'][:18] + "..." if len(row['title']) > 18 else row['title']
                    st.markdown(f"**{display_title}**")
                    st.markdown(f"⭐ **{round(final_rating, 1)}/10**")
                    
                    with st.popover("Overview"):
                        st.write(row.get('overview', 'No description available'))
    else:
        st.error("Movie not found!")
