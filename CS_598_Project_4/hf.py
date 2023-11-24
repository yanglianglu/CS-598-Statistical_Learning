import numpy as np
import pandas as pd
import streamlit as st
import requests
import scipy.sparse as sparse
from pathlib import Path


@st.cache_data
def load_data():
    # use absolute path
    dir = Path(__file__).parent.absolute()

    rating_path = dir / 'data/ratings.dat'
    movie_path = dir / 'data/movies.dat'
    print(movie_path)

    ratings = pd.read_csv(rating_path, sep='::', engine='python', header=None, )
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    movies = pd.read_csv(fr'{movie_path}', sep='::', engine='python', header=None, encoding="ISO-8859-1")
    movies.columns = ['MovieID', 'Title', 'Genres']
    return ratings, movies


@st.cache_data
def genres_matrix(movies):
    # Create a genres matrix
    genres = []
    for i in range(len(movies)):
        genres.extend(movies.iloc[i]['Genres'].split('|'))
    genres = list(set(genres))
    genres.sort()
    genres_matrix = pd.DataFrame(columns=genres)
    for i in range(len(movies)):
        genres_matrix.loc[i] = [0] * len(genres)
        for genre in movies.iloc[i]['Genres'].split('|'):
            genres_matrix.loc[i][genre] = 1
    genres_matrix.index = movies['MovieID']
    return genres_matrix


@st.cache_data
def subset_movies(ratings_df, genre_df, genre):
    # giving a genre, return a subset of movies that are in that genre
    genre_movies = genre_df[genre_df[genre] == 1]

    genre_index = genre_movies.index
    ratings_df = ratings_df[ratings_df['MovieID'].isin(genre_index)]
    rating_matrix = pd.pivot_table(ratings_df, values='Rating', index=['UserID'],
                                   columns=['MovieID'])
    return rating_matrix


def most_watched_movies(genre_rating):
    # return a list of most watched movies
    # input is a rating matrix
    # output is a list of movie ids
    movie_count = genre_rating.count()
    movie_count = movie_count
    # scale to 0 to 1
    movie_count = movie_count / movie_count.max()
    movie_count = movie_count.rename('popularity')
    return movie_count


def highly_rated_movies(genre_rating):
    # return a list of highly rated movies
    # input is a rating matrix
    # output is a list of movie ids
    # using Weighted Rating (WR) = (v / (v+m)) × R + (m / (v+m)) × C
    mean = genre_rating.mean()
    v = genre_rating.count()
    m = 300
    C = mean.mean()
    wr = (v / (v + m)) * mean + (m / (v + m)) * C
    # scale to 0 to 1
    wr = wr / wr.max()
    wr = wr.rename('rating')
    return wr


def ranking(genre_rating, n):
    # return a list of movies that are ranked by most watched and highly rated
    # input is a rating matrix
    # output is a list of movie ids
    movie_count = most_watched_movies(genre_rating)
    wr = highly_rated_movies(genre_rating)
    rank = pd.merge(movie_count, wr, left_index=True, right_index=True)

    # popularity count 1/3, rating 2/3
    rank['score'] = rank['popularity'] / 3 + rank['rating'] * 2 / 3
    # scale
    rank['score'] = rank['score'] / rank['score'].max()
    return rank.sort_values(by='score', ascending=False).head(n)


def get_recommendations(movies, rating, genres, genre, n):
    # return a list of movie ids that are recommended
    # input is a rating matrix, genres matrix, genre, and number of recommendations
    # output is a list of movie ids
    genre_rating = subset_movies(rating, genres, genre)
    rank = ranking(genre_rating, n)
    index = rank.index
    recommend_movie = movies[movies['MovieID'].isin(index)]
    recommend_movie = parse_columns(recommend_movie)
    return recommend_movie


def parse_columns(df):
    # convert Genres to list
    df.loc[:, 'Genres'] = df['Genres'].str.split('|')
    # split title and year
    split = df['Title'].str.split('(')
    title = split.str[0].copy()
    year = split.str[1].copy().str.replace(')', '')
    df.loc[:, 'Title'] = title
    df.loc[:, 'Year'] = year
    # reindex
    df.index = range(1, len(df) + 1)
    return df


def get_movie_info(movie_name, year):
    # input is a movie id
    # output is a list of movie info
    api_key = open('data/api_key').read()
    data_url = 'http://www.omdbapi.com/?apikey=' + api_key + '&'
    poster_url = 'http://img.omdbapi.com/?apikey=' + api_key + '&'
    data_url = data_url + 't=' + movie_name + '&y=' + year
    data = requests.get(data_url).json()
    if data.get('Response') == 'False':
        return None
    imbdid = data.get('imdbID')
    poster_url = poster_url + 'i=' + imbdid
    poster = requests.get(poster_url)
    # check if 404 response
    if poster.status_code == 404:
        from PIL import Image
        import io
        img = Image.new('RGB', (100, 100), color=(73, 109, 137))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        poster = img_byte_arr
        data['Poster'] = poster
    else:
        data['Poster'] = poster.content
    return data


def create_movie_display(movie):
    with st.container():
        # Use columns to layout the poster and details side by side
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(movie['Poster'], use_column_width=True)
        with col2:
            with st.expander(f"{movie['Title']} ({movie['Year']})", expanded=False):
                st.write(f"**Director:** {movie['Director']}")
                st.write(f"**Actors:** {movie['Actors']}")
                st.write(f"**Genres:** {movie['Genre']}")  # Join genres with a comma
                st.write(f"**Runtime:** {movie['Runtime']}")
                st.write(" ")
                st.write(f"**Plot:** {movie['Plot']}")
                st.write(f"**Awards:** {movie['Awards']}")
                st.write(" ")
                st.write("**Ratings:**")
                # Display ratings in a more structured way
                for rating in movie['Ratings']:
                    st.write(f"{rating['Source']}: {rating['Value']}")


@st.cache_data
def get_best_movies(movies, ratings, genres):
    # return a dataframe of the best movies from each genre
    # input is a rating matrix, genres matrix
    best = pd.DataFrame()
    for genre in genres.columns:
        best = pd.concat([best, get_recommendations(movies, ratings, genres, genre, 3)])
    best.index = range(1, len(best) + 1)
    # drop duplicates
    best = best.drop_duplicates(subset=['Title']).reset_index(drop=True)
    return best


@st.cache_data
def read_sparse_matrix():
    # read sparse matrix
    dir = Path(__file__).parent.absolute()
    sparse_matrix = sparse.load_npz(str(dir) + '/data/similarity_matrix.npz')
    return sparse_matrix


def create_vector(movies_with_ratings):
    # create a vector of user ratings
    # input is a list of user ratings
    # output is a vector of user ratings
    # initialize a vector of 0
    movies_with_ratings = movies_with_ratings.set_index('MovieID')['Rating']
    movies_with_ratings = movies_with_ratings.reindex(range(1, 3707), fill_value=0)
    return movies_with_ratings


def recommend_movies(new_user_ratings, similarity_sparse, n_recommendations=10):
    """
    Generate movie recommendations based on new user ratings and a sparse similarity matrix.

    Parameters:
    - new_user_ratings: np.array, user's ratings for movies; 0 indicates the movie hasn't been rated.
    - similarity_sparse: scipy.sparse matrix, item-item similarity matrix in sparse format.
    - n_recommendations: int, the number of recommendations to return.

    Returns:
    - List of movie indices representing the top N recommendations.
    """

    # Validate the shape of new_user_ratings
    if new_user_ratings.shape[0] != similarity_sparse.shape[0]:
        raise ValueError("The length of new_user_ratings must match the size of similarity matrix.")

    # Convert new user ratings to NaN if 0 (user hasn't rated the movie)
    user_ratings = np.where(new_user_ratings == 0, np.nan, new_user_ratings)

    # Filter out movies the user has already rated
    unrated_movies_mask = np.isnan(user_ratings)

    # Extract the similarity scores for unrated movies
    unrated_similarity = similarity_sparse[unrated_movies_mask, :]

    # Calculate the weighted scores using matrix multiplication
    weighted_scores = unrated_similarity.dot(user_ratings)

    # Normalize by the sum of the similarities for rated movies
    sum_similarity = unrated_similarity.sum(axis=1).A1  # Convert to 1D array
    valid_mask = sum_similarity > 0
    normalized_scores = np.divide(weighted_scores, sum_similarity, where=valid_mask)

    # Select top N recommendations
    top_movie_indices = np.argsort(-normalized_scores)[:n_recommendations]

    return top_movie_indices.tolist()


def get_movie_titles(movies, movie_ids):
    sub = movies[movies['MovieID'].isin(movie_ids)]
    sub = parse_columns(sub)
    return sub
