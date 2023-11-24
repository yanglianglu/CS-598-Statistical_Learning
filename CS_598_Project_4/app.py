import streamlit as st
import json
import pandas as pd
from hf import *
import random
ratings, movies = load_data()
genres = genres_matrix(movies)
genre_list = json.loads(open('data/genres.json').read())
best_movies = get_best_movies(movies, ratings, genres)
similarity_matrix = read_sparse_matrix()
# Set the title of the app
st.sidebar.title("Movie Recommender")

# Add sidebar options
genre = st.sidebar.radio("Choose a recommendation type:", ('Recommender by Genre', 'Recommender by Rating'))

# Main content of the app
st.header("Movie Recommender")

# Step 1: Select your favorite genre
st.subheader('Step 1: Select your favorite genre')
selected_genre = st.selectbox("Select a single genre from the dropdown menu", genre_list)

# Step 2: Discover movies you might like
st.subheader('Step 2: Discover movies you might like')
if genre == 'Recommender by Genre':
    if st.button('Click here to get your recommendations'):
        genre_rating = get_recommendations(movies, ratings, genres, selected_genre, 10)
        st.write('Here are some movies you might like:')
        for i, movie in genre_rating.iterrows():
            info = get_movie_info(movie['Title'], movie['Year'])
            if info is not None:
                create_movie_display(info)
        # get movie info
if genre == 'Recommender by Rating':

    st.write("Please rate the following movies:")

    # Create a list of random movie ids from the best movies dataframe
    random_movies = random.sample(best_movies.index.tolist(), 10)
    subset_movies = best_movies.loc[random_movies]

    # Create a list of movie ids from the best movies dataframe
    # Initialize session state variables if not already present
    current_index = st.session_state.get('current_index', 0)
    st.session_state.rating = st.session_state.get('rating', [])
    if current_index < len(subset_movies) and current_index < 10:  # Limit to 10 movies
        movie = subset_movies.iloc[current_index]
        with st.chat_message("user"):
            st.write(f"Movie: {movie['Title']} ({movie['Year']})")
            # Use a form to capture the slider interaction before rerun
            with st.form(key=f'rating_form_{current_index}'):
                info = get_movie_info(movie['Title'], movie['Year'])
                if info is not None:
                    create_movie_display(info)
                rating = st.slider("Your Rating:", 1, 5)

                # create a checkbox if you want to skip the movie
                skip = st.checkbox("Skip")
                submitted = st.form_submit_button("Submit Rating")
                if submitted:
                    # Save the rating and increment the index
                    if skip:
                        rating = 0
                    st.session_state.rating.append(rating)
                    st.session_state.current_index = current_index + 1
                    # Display the next movie or end
                    st.rerun()
    else:
        subset_movies.loc[:, 'Rating'] = st.session_state.rating
        user_vector = create_vector(subset_movies)
        recommendations = recommend_movies(user_vector, similarity_matrix, 10)
        recommended_movies_df = get_movie_titles(movies, recommendations)

        st.write("You have rated 10 movies. Here are your recommendations:")
        for i, movie in recommended_movies_df.iterrows():
            info = get_movie_info(movie['Title'], movie['Year'])
            if info is not None:
                with st.container():
                    st.write(f"{movie['Title']} ({movie['Year']})")

                    create_movie_display(info)

# To run the app, save this script and run `streamlit run app.py` from the terminal.
