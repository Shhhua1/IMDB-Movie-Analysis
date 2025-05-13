import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Read the data from the CSV file
try:
    df = pd.read_csv('imdb_cleaned_data.csv')
except FileNotFoundError:
    st.error("The file 'imdb_top_1000.csv' was not found. Please ensure it exists in the same directory as this script.")
    st.stop()
    

class MovieRecommenderApp:
    def __init__(self, df):
        self.df = df.copy()
        self.tfidf = TfidfVectorizer(
            stop_words='english',
            max_df=0.8,
            min_df=2,
            ngram_range=(1, 2)
        )
        self.prepare_features()
        self.user_movie_list = pd.DataFrame()
        self.tfidf_matrix = self.tfidf.fit_transform(self.df['combined_features'])
        self.cosine_sim = linear_kernel(self.tfidf_matrix, self.tfidf_matrix)
        self.indices = pd.Series(self.df.index, index=self.df['Series_Title']).drop_duplicates()

    def prepare_features(self):
        self.df['combined_features'] = (
            self.df['Overview'].fillna('') + ' ' +
            self.df['Genre'].fillna('') + ' ' +
            self.df['Director'].fillna('')
        )
        self.df['combined_features'] = self.df['combined_features'].fillna('')

    def autocomplete_movie_name(self):
        
        user_selection = st.multiselect(
            'Select Movie Name',
            self.df['Series_Title'].unique()
        )
        if user_selection:
            return self.df[self.df['Series_Title'].isin(user_selection)].copy()
        # Return an empty DataFrame with the same columns as self.df
        return pd.DataFrame(columns=self.df.columns)
    
    def get_recommendations(self, selected_movies_df, num_recommendations=10):
        idx = self.indices[selected_movies_df['Series_Title'].values[0]]
        sim_scores = list(enumerate(self.cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_similar = sim_scores[1:num_recommendations+1]
        movie_indices = [i[0] for i in top_similar]
        similarity_scores = [i[1] for i in top_similar]
        results = pd.DataFrame({
            'Recommended Movie': self.df['Series_Title'].iloc[movie_indices].values,
            'Similarity Score': similarity_scores
        })
        return results

    def run(self):
        selected_movies_df = self.autocomplete_movie_name()
        if not selected_movies_df.empty:
            st.subheader('Selected Movies')
            st.write(selected_movies_df)
            st.subheader('Recommended Movies')
            recommended_movies = self.get_recommendations(selected_movies_df)
            st.write(recommended_movies)

app = MovieRecommenderApp(df)
app.run()