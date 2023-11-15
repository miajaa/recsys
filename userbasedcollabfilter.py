import pandas as pd
import numpy as np

ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
num_ratings = len(ratings)


# B) Implement user-based collaborative filtering with Pearson correlation
def compute_user_similarity(user_id):
    # Filtering ratings for a specific user and setting 'movie_id' as the index
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    similarity_scores = {}

    # Loop through each user in the dataset
    for other_user_id, group in ratings.groupby('user_id'):
        # Skipping the current user
        if other_user_id != user_id:
            # Setting 'movie_id' as the index for the other user's ratings
            other_user_ratings = group.set_index('movie_id')['rating']
            # Finding common movies
            common_movies = user_ratings.index.intersection(other_user_ratings.index)

            # If there are at least 5 common movies
            if len(common_movies) >= 5:
                user_ratings_common = user_ratings[common_movies]
                other_user_ratings_common = other_user_ratings[common_movies]

                # Checking for non-zero standard deviations
                if user_ratings_common.std() != 0 and other_user_ratings_common.std() != 0:
                    # Compute Pearson correlation coefficient and store it in the dictionary
                    similarity_scores[other_user_id] = round(
                        np.corrcoef(user_ratings_common, other_user_ratings_common)[0, 1], 3)

    return similarity_scores


# C) Predict movie scores
def predict_movie_score(user_id, movie_id, similar_users):
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    total_similarity = 0
    weighted_sum = 0

    # Filter ratings for similar users who have rated the specified movie
    other_user_ratings = \
        ratings[(ratings['user_id'].isin(similar_users.keys())) & (ratings['movie_id'] == movie_id)].set_index(
            'user_id')[
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
    return round(weighted_sum / total_similarity, 3)


# D) Select a user from the dataset, and for this user, show the 10 most similar users and the 10 most relevant movies
def get_top_similar_users(user_id):
    # Calculate similar users for the specified user
    similar_users = compute_user_similarity(user_id)
    # Get the top 10 similar users
    top_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_similar_users  # Return the list of top similar users


def get_top_recommended_movies(user_id, similar_users):
    user_rated_movies = ratings[ratings['user_id'] == user_id]['movie_id']
    all_movies = set(ratings['movie_id'])
    unrated_movies = list(all_movies - set(user_rated_movies))

    top_movies = []
    for movie_id in unrated_movies:
        # Predict the movie score
        prediction = predict_movie_score(user_id, movie_id, similar_users)
        if prediction is not None:
            # Add movie ID and prediction to the list
            top_movies.append((movie_id, round(prediction, 3)))

    # Get the top 10 recommended movies
    top_recommended_movies = sorted(top_movies, key=lambda x: x[1], reverse=True)[:10]

    # Return the list of top recommended movies
    return top_recommended_movies


# E) Design and implement a new similarity function
def compute_cosine_similarity(user_id):
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    similarity_scores = {}

    for other_user_id, group in ratings.groupby('user_id'):
        # Skip the current user
        if other_user_id != user_id:
            other_user_ratings = group.set_index('movie_id')['rating']
            common_movies = user_ratings.index.intersection(other_user_ratings.index)

            # Ensure at least 10 common movies for meaningful similarity
            if len(common_movies) >= 2:
                user_ratings_common = user_ratings[common_movies]
                other_user_ratings_common = other_user_ratings[common_movies]

                # Compute cosine similarity
                similarity = np.dot(user_ratings_common, other_user_ratings_common) / (
                        np.linalg.norm(user_ratings_common) * np.linalg.norm(other_user_ratings_common))

                similarity_scores[other_user_id] = round(similarity, 3)

    return similarity_scores


# F) Implement group recommendation using user-based collaborative filtering and aggregation methods

# F.1 Average Aggregation Method
def group_recommendation_average(user_ids, movie_id):
    # Calculate predictions for each user in the group
    individual_ratings = []
    for user_id in user_ids:
        similar_users = compute_user_similarity(user_id)
        prediction = predict_movie_score(user_id, movie_id, similar_users)

        # If prediction is available, add it to the list
        if prediction is not None:
            individual_ratings.append(prediction)

    # If no predictions are available, return None
    if not individual_ratings:
        return None

    # Calculate the average rating across all users
    average_rating = np.mean(individual_ratings)
    return round(average_rating, 3)


# F.2 Least Misery Aggregation Method
def group_recommendation_least_misery(user_ids, movie_id):
    # Calculate predictions for each user in the group
    individual_ratings = []
    for user_id in user_ids:
        similar_users = compute_user_similarity(user_id)
        prediction = predict_movie_score(user_id, movie_id, similar_users)

        # If prediction is available, add it to the list
        if prediction is not None:
            individual_ratings.append(prediction)

    # If no predictions are available, return None
    if not individual_ratings:
        return None

    # Calculate the least misery rating (minimum rating across all users)
    least_misery_rating = np.min(individual_ratings)
    return round(least_misery_rating, 3)


# F.7 Function to generate group of 3 users
def generate_group_of_users():
    group_members = []
    for i in range(3):  # You can change the number of group members as needed
        while True:
            try:
                member_id = int(input(f"Enter user ID for group member {i + 1}: "))
                if member_id not in ratings['user_id'].unique():
                    print("Invalid user ID. Please enter a valid user ID.")
                else:
                    group_members.append(member_id)
                    break
            except ValueError:
                print("Invalid input. Please enter a valid user ID.")
    return group_members


# F.8 Generate group of 3 users
group_members = generate_group_of_users()

# F.9 Prompt user for movie ID for group recommendation
while True:
    try:
        group_movie_id = int(input("Enter movie ID for group recommendation: "))
        if group_movie_id not in ratings['movie_id'].unique():
            print("Invalid movie ID. Please enter a valid movie ID.")
        else:
            break
    except ValueError:
        print("Invalid input. Please enter a valid movie ID.")

# F.10 Generate group recommendations using both aggregation methods
group_recommendation_avg = group_recommendation_average(group_members, group_movie_id)
group_recommendation_lm = group_recommendation_least_misery(group_members, group_movie_id)

# F.11 Display group recommendations
print(f'\nGroup Recommendations for Movie {group_movie_id} Using Average Aggregation: {group_recommendation_avg}')
print(f'Group Recommendations for Movie {group_movie_id} Using Least Misery Aggregation: {group_recommendation_lm}')

# F.12 Show top 10 movie recommendations for the group
print("\nTop 10 Movie Recommendations:")
# Average Aggregation
top_movies_avg = []
for user_id in group_members:
    similar_users = compute_user_similarity(user_id)
    top_movies_avg.extend(get_top_recommended_movies(user_id, similar_users))
top_movies_avg = sorted(set(top_movies_avg), key=lambda x: x[1], reverse=True)[:10]
print(f'Using Average Aggregation: {top_movies_avg}')

# Least Misery Aggregation
top_movies_lm = []
for user_id in group_members:
    similar_users = compute_user_similarity(user_id)
    top_movies_lm.extend(get_top_recommended_movies(user_id, similar_users))
top_movies_lm = sorted(set(top_movies_lm), key=lambda x: x[1], reverse=True)[:10]
print(f'Using Least Misery Aggregation: {top_movies_lm}')


# F.13 Disagreements-Aware Aggregation Method
def group_recommendation_disagreements(user_ids, movie_id):
    # Calculate predictions for each user in the group
    individual_ratings = {}
    for user_id in user_ids:
        similar_users = compute_user_similarity(user_id)
        prediction = predict_movie_score(user_id, movie_id, similar_users)

        # If prediction is available, add it to the dictionary
        if prediction is not None:
            individual_ratings[user_id] = prediction

    # If no predictions are available, return None
    if not individual_ratings:
        return None

    # Calculate the standard deviation of ratings across all users
    disagreements = np.std(list(individual_ratings.values()))
    weighted_avg_disagreements = np.mean(list(individual_ratings.values())) - disagreements

    return round(weighted_avg_disagreements, 3)


# F.14 Generate group recommendations using disagreements-aware aggregation method
group_recommendation_disagreements = group_recommendation_disagreements(group_members, group_movie_id)

# F.15 Display group recommendations with disagreements-aware aggregation
print(
    f'\nGroup Recommendations for Movie {group_movie_id} Using Disagreements-Aware Aggregation: {group_recommendation_disagreements}')

# F.16 Show top 10 movie recommendations for the group with disagreements-aware aggregation
top_movies_disagreements = []
for user_id in group_members:
    similar_users = compute_user_similarity(user_id)
    top_movies_disagreements.extend(get_top_recommended_movies(user_id, similar_users))
top_movies_disagreements = sorted(set(top_movies_disagreements), key=lambda x: x[1], reverse=True)[:10]
print(f'Using Disagreements-Aware Aggregation, the 10 top movies are: {top_movies_disagreements}')
