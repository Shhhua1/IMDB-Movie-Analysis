import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
#import plotly.graph_objects as go
from Prediction_algorithm import MovieRecommenderApp
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import linear_kernel




# Read the data from the CSV file
try:
    df = pd.read_csv('imdb_cleaned_data.csv')
except FileNotFoundError:
    st.error("The file 'imdb_top_1000.csv' was not found. Please ensure it exists in the same directory as this script.")
    st.stop()
    
#st.set_page_config(layout="wide")

def page_home():
    """Display the home page with a welcome message."""
    st.markdown("<h1 style='text-align: center;'>Welcome!</h1>", unsafe_allow_html=True)
    st.write('---')
    st.markdown("<h5 style='text-align: center;'>This app does things</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>Use the sidebar on the left to get started.</h5>", unsafe_allow_html=True)

def main():
    """Main function to run the Streamlit app."""
    st.title("Movie Recommender System")
    st.write("Select a movie to get recommendations.")
    
    # Create an instance of the MovieRecommenderApp
    recommender = MovieRecommenderApp(df)
    
    # Autocomplete movie name
    selected_movies_df = recommender.autocomplete_movie_name()
    
    if not selected_movies_df.empty:
        # Get recommendations
        recommendations = recommender.get_recommendations(selected_movies_df)
        
        # Display recommendations
        st.write("Recommended Movies:")
        st.dataframe(recommendations)


page = st.sidebar.selectbox(
    'Select a page',
    ('Home', 'Movie Recommender')
)

if page == 'Home':
    page_home()
elif page == 'Movie Recommender':
    main()
