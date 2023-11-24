import pandas as pd
import numpy as np

# A) Read from file
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
num_ratings = len(ratings)
movie_list = ratings['movie_id'].unique()


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

            # If there are at least 2 common movies
            if len(common_movies) >= 2:
                user_ratings_common = user_ratings[common_movies]
                other_user_ratings_common = other_user_ratings[common_movies]

                # Checking for non-zero standard deviations
                if user_ratings_common.std() != 0 and other_user_ratings_common.std() != 0:
                    # Compute Pearson correlation coefficient and store it in the dictionary
                    similarity_scores[other_user_id] = round(
                        np.corrcoef(user_ratings_common, other_user_ratings_common)[0, 1], 3)

    return similarity_scores


# C) Predict movie scores if rating doesn't exist
def predict_movie_score(user_id, movie_id, similar_users, floor=0.5):   # Ratings can't be less than 0.5
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    total_similarity = 0
    weighted_sum = 0
    predicted_score = 0
    
    if movie_id not in user_ratings.index:
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

        # Calculate the predicted movie score
        predicted_score = weighted_sum / total_similarity

        # Negative movie scores aren't allowed, instead apply the floor value
        predicted_score = max(predicted_score, floor)

    if movie_id in user_ratings.index:
        predicted_score = user_ratings[movie_id]

    return predicted_score


# D) Select a user from the dataset, and for this user, show the 10 most similar users and the 10 most relevant movies
def get_top_similar_users(user_id):
    # Calculate similar users for the specified user
    similar_users = compute_user_similarity(user_id)
    # Get the top 10 similar users
    top_similar_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:10]
    return top_similar_users  # Return the list of top similar users


# Generate a list of recommended movies for one user
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
def group_recommendation_average(individual_ratings):
    # Extract values
    ratings_values = np.array(list(individual_ratings.values()))
    # Calculate the average rating across all users
    average_rating = np.mean(ratings_values)
    return round(average_rating, 3)


# F.2 Least Misery Aggregation Method
def group_recommendation_least_misery(individual_ratings):
    # Extract values
    ratings_values = np.array(list(individual_ratings.values()))
    # Calculate the least misery rating (minimum rating across all users)
    least_misery_rating = np.min(ratings_values)
    return round(least_misery_rating, 3)


# F.3 Disagreements-Aware Aggregation Method
def group_recommendation_disagreements(individual_ratings, coefficient=0.2):
    # Calculate the standard deviation of ratings across all users
    disagreements = np.std(list(individual_ratings.values()))

    # Handle the case where there is only one user in the group or there's no disagreements
    if len(individual_ratings) == 1 or disagreements == 0:
        return round(list(individual_ratings.values())[0], 3)

    # Calculate the disagreements-aware aggregation
    weighted_avg_disagreements = np.mean(list(individual_ratings.values())) + coefficient * disagreements
    return round(weighted_avg_disagreements, 3)


# F.4 Function to generate group of 3 users
def generate_group_of_users(num_members=3): # number of members can be specified
    group_members = []

    for i in range(num_members):
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


# F.5 Calculate or fetch rating for given movie for each member of the group
def calculate_user_ratings(user_ids, movie_id):
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
    return individual_ratings


# F.6 Calculate top 10 recommendations with four different aggregation methods
def recommend_movies(movie_list, member_ids, aggregation_method, dynamic_weights=None, coefficient=0.):
    # Dictionary to store aggregated scores for each movie
    aggregated_scores = {}

    # Store user similarities for all members and find existing ratings
    all_similarities = {}
    all_user_ratings = {}

    for member_id in member_ids:
        all_similarities[member_id] = compute_user_similarity(member_id)
        all_user_ratings[member_id] = ratings[ratings['user_id'] == member_id].set_index('movie_id')['rating']

    # Loop through each movie in the list
    for movie_id in movie_list:
        # Reset member_scores for the next movie
        member_scores = []
        # Loop through each group member
        for member_id in member_ids:
            similar_users = all_similarities[member_id]
            user_ratings = all_user_ratings[member_id]

            # Check if the user has a rating for the movie
            if movie_id not in user_ratings.index:
                # If not, predict the rating using the predict_movie_score function
                prediction = predict_movie_score(member_id, movie_id, similar_users)

                # Add the movie_id to user_ratings with the predicted score
                user_ratings.at[movie_id] = prediction if prediction is not None else 0  # Use 0 as a default if prediction is None

            # Add the score
            member_scores.append(user_ratings[movie_id])

        # Calculate disagreement for each movie
        disagreement = np.std(member_scores)

        # Check if any predictions were available
        if member_scores:
            # Apply the aggregation method
            if aggregation_method == 'Average':
                aggregated_score = round(np.mean(member_scores), 3)
            elif aggregation_method == 'Least Misery':
                aggregated_score = round(np.min(member_scores), 3)
            # Apply the disagreement-aware aggregation
            elif aggregation_method == 'Disagreement Aware':
                aggregated_score = round(np.mean(member_scores) + coefficient * disagreement, 3)
            # Apply the dynamic least misery aggregation
            elif aggregation_method == 'Dynamic Least Misery' and dynamic_weights is not None:
                aggregated_score = round(np.sum(dynamic_weights * member_scores), 3)

            # Add the aggregated score to the dictionary
            aggregated_scores[movie_id] = aggregated_score

    # Sort and display the top 10 movies
    top_movies = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f'{aggregation_method} Aggregation: {top_movies}')
    return top_movies


# F.7 Calculate dynamic weights based on least misery scores
def calculate_dynamic_weights_least_misery(previous_recommendations, member_ids, coefficient=0.4):
    # Dictionary to store the mean least misery scores for each user
    mean_least_misery_scores = {}

    # Check if there are previous recommendations
    if previous_recommendations is not None:
        # Calculate least misery scores for each user and each movie in the previous recommendations
        for member_id in member_ids:
            member_scores = []
            for movie_id, _ in previous_recommendations:
                similar_users = compute_user_similarity(member_id)
                user_ratings = ratings[ratings['user_id'] == member_id].set_index('movie_id')['rating']

                # Check if the user has a rating for the movie
                if movie_id not in user_ratings.index:
                    # If not, predict the rating using the predict_movie_score function
                    prediction = predict_movie_score(member_id, movie_id, similar_users)

                    # Add the movie_id to user_ratings with the predicted score
                    user_ratings.at[movie_id] = prediction if prediction is not None else 0  # Use 0 as a default if prediction is None

                # Add the score
                member_scores.append(user_ratings[movie_id])

            # Calculate mean least misery score for the user
            mean_least_misery_scores[member_id] = np.mean(sorted(member_scores)[:10])

    # If no previous recommendations, initialize with equal weights
    else:
        for member_id in member_ids:
            mean_least_misery_scores[member_id] = 1 / len(member_ids)

    # Calculate dynamic weights based on mean least misery scores
    dynamic_weights = np.array([1 / (1 + coefficient * score) for score in mean_least_misery_scores.values()])

    # Normalize dynamic weights
    dynamic_weights /= np.sum(dynamic_weights)

    return dynamic_weights


### PRINTS AND FUNCTION CALLS ###

# Generate group of 3 users
group_members = generate_group_of_users()

# F.8 Show top 10 movie recommendations for the group using dynamic weights
print('\nTop 10 Sequential Movie Recommendations with Dynamic Weights:')
rounds = 3
previous_weights = np.ones(len(group_members)) / len(group_members)
previous_recommendations = None
for i in range(rounds):

    # Use previous weights to adjust dynamic weights
    if previous_weights is not None:
        dynamic_weights = calculate_dynamic_weights_least_misery(previous_recommendations, group_members)

    # Display recommendations
    print(f'\nGroup Recommendation Using Dynamic Least Misery Aggregation (Round {i + 1}):')
    print(f'Dynamic Weights: {dynamic_weights}')
    previous_recommendations = recommend_movies(movie_list, group_members, 'Dynamic Least Misery', dynamic_weights)
    
    # Update the dynamic weights for the next round
    previous_weights = dynamic_weights