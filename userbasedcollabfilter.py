import pandas as pd  # Importing the pandas library and aliasing it as 'pd'
import numpy as np  # Importing the numpy library and aliasing it as 'np'

# A) Load the dataset
# Reading a CSV file into a DataFrame and assigning it to the variable 'ratings'
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])  #
print(ratings.head())

num_ratings = len(ratings)
print(f'Total number of ratings: {num_ratings}')


# B) Implement user-based collaborative filtering with Pearson correlation

def compute_user_similarity(user_id):
    # Get the ratings of the specified user and set the movie_id as the index
    # Filtering ratings for a specific user and setting 'movie_id' as the index
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    similarity_scores = {}

    # Loop through each user in the dataset
    # Grouping the data by 'user_id'
    for other_user_id, group in ratings.groupby('user_id'):

        # Skipping the current user
        if other_user_id != user_id:
            # Setting 'movie_id' as the index for the other user's ratings
            other_user_ratings = group.set_index('movie_id')['rating']
            # Finding common movies
            common_movies = user_ratings.index.intersection(other_user_ratings.index)

            # If there are at least 2 common movies
            if len(common_movies) >= 2:
                # Get ratings for common movies for the specified user
                user_ratings_common = user_ratings[common_movies]
                # Get ratings for common movies for the other user
                other_user_ratings_common = other_user_ratings[common_movies]
                # Checking for non-zero standard deviations

                if user_ratings_common.std() != 0 and other_user_ratings_common.std() != 0:
                    # Compute Pearson correlation coefficient and store it in the dictionary
                    similarity_scores[other_user_id] = np.corrcoef(user_ratings_common, other_user_ratings_common)[0, 1]

    return similarity_scores


# C) Predict movie scores

def predict_movie_score(user_id, movie_id, similar_users):
    # Get the ratings of the specified user and set the movie_id as the index
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    total_similarity = 0  # Initializing a variable to store the total similarity
    weighted_sum = 0  # Initializing a variable to store the weighted sum

    # Filter ratings for similar users who have rated the specified movie
    other_user_ratings = \
    ratings[(ratings['user_id'].isin(similar_users.keys())) & (ratings['movie_id'] == movie_id)].set_index('user_id')[
        'rating']

    # Loop through common users and calculate weighted sum and total similarity
    for user in user_ratings.index.intersection(other_user_ratings.index):
        similarity = similar_users[user]
        user_rating = user_ratings[user]
        other_user_rating = other_user_ratings[user]

        weighted_sum += similarity * other_user_rating
        total_similarity += abs(similarity)

    # If total similarity is zero (to avoid division by zero)
    if total_similarity == 0:
        return None

    # Return the predicted movie score
    return weighted_sum / total_similarity


# Example usage:
user_id = int(input("Enter user ID: "))  # Prompting user for input
movie_id = int(input("Enter movie ID: "))  # Prompting user for input

# Calculate similar users for the specified user
similar_users = compute_user_similarity(user_id)

# Predict the movie score
prediction = predict_movie_score(user_id, movie_id, similar_users)
print(f'Predicted rating for movie {movie_id} by user {user_id}: {prediction}')


# (D) Select a user from the dataset, and for this user, show the 10 most similar users and the 10 most relevant movies
def get_top_similar_users(user_id):
    # Calculate similar users for the specified user
    similar_users = compute_user_similarity(user_id)
    # Get the top 10 similar users
    top_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_similar_users  # Return the list of top similar users


def get_top_recommended_movies(user_id, similar_users):
    # Get the movies rated by the specified user
    user_rated_movies = ratings[ratings['user_id'] == user_id]['movie_id']
    # Get all unique movie IDs
    all_movies = set(ratings['movie_id'])
    # Get unrated movies
    unrated_movies = list(all_movies - set(user_rated_movies))

    top_movies = []
    for movie_id in unrated_movies:
        # Predict the movie score
        prediction = predict_movie_score(user_id, movie_id, similar_users)
        if prediction is not None:
            # Add movie ID and prediction to the list
            top_movies.append((movie_id, prediction))

            # Get the top 10 recommended movies
    top_recommended_movies = sorted(top_movies, key=lambda x: x[1], reverse=True)[:10]

    # Return the list of top recommended movies
    return top_recommended_movies


# E) Design and implement a new similarity function
def compute_cosine_similarity(user_id):
    # Get the ratings of the specified user and set the movie_id as the index
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    similarity_scores = {}  # Initialize a dictionary to store similarity scores

    for other_user_id, group in ratings.groupby('user_id'):
        if other_user_id != user_id:  # Skip the current user
            # Set the movie_id as the index for the other user's ratings
            other_user_ratings = group.set_index('movie_id')['rating']
            common_movies = user_ratings.index.intersection(other_user_ratings.index)

            if len(common_movies) >= 2:  # Ensure at least 2 common movies for meaningful similarity
                user_ratings_common = user_ratings[common_movies]
                other_user_ratings_common = other_user_ratings[common_movies]

                # Compute cosine similarity
                similarity = np.dot(user_ratings_common, other_user_ratings_common) / (
                            np.linalg.norm(user_ratings_common) * np.linalg.norm(other_user_ratings_common))

                similarity_scores[other_user_id] = similarity  # Store the similarity score

    return similarity_scores  # Return the dictionary of similarity scores


# Define a user ID
selected_user_id = int(input("Enter selected user ID: "))  # Prompting user for input

# Get the top similar users for the selected user
top_similar_users = get_top_similar_users(selected_user_id)

print(f'Top 10 most similar users to User (user_id, similarity score): {selected_user_id}: {top_similar_users}')

top_recommended_movies = get_top_recommended_movies(selected_user_id, similar_users)

print(f'Top 10 recommended movies for User (movie_id, predicted score): {selected_user_id}: {top_recommended_movies}')

# Example usage for the new similarity function
user_id = int(input("Enter user ID for new similarity: "))  # Prompting user for input

# Calculate similar users using the new similarity function
similar_users_new = compute_cosine_similarity(user_id)

# Get the top 10 similar users using the new similarity function
top_similar_users_new = sorted(similar_users_new.items(), key=lambda x: x[1], reverse=True)[:10]

print(f'Top 10 most similar users to User {user_id} (user_id, similarity score): {top_similar_users_new}')
