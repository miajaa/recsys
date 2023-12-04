import pandas as pd
import numpy as np

# A) Read from file
ratings = pd.read_csv('u.data', sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
num_ratings = len(ratings)
movie_list = ratings['movie_id'].unique()

# Read the content of the u.genre file
with open('u.genre', 'r') as file:
    genre_content = file.readlines()

# Create a dictionary to store genre information
genre_dict = {
    'unknown': [],
    'Action': [],
    'Adventure': [],
    'Animation': [],
    'Children\'s': [],
    'Comedy': [],
    'Crime': [],
    'Documentary': [],
    'Drama': [],
    'Fantasy': [],
    'Film-noir': [],
    'Horror': [],
    'Musical': [],
    'Mystery': [],
    'Romance': [],
    'Sci-fi': [],
    'Thriller': [],
    'War': [],
    'Western': []
}

# Iterate through each line and populate the dictionary
for line in genre_content:
    parts = line.strip().split('|')  # Use '|' as the delimiter for tab-separated values
    genre_name = parts[0]

    # Ensure there are at least two parts before accessing the second part
    if len(parts) >= 2:
        genre_dict[genre_name] = []  # Create an empty list for each genre

# Read the content of the u.item file
with open('u.item', 'r', encoding='latin-1') as file:
    movies_content = file.readlines()

# Create a dictionary to store movie information
movies_dict = {}

# Iterate through each line in the u.item file
for line in movies_content:
    parts = line.strip().split('|')
    movie_title_with_year = parts[1]
    movie_id = int(parts[0])

    # Remove the last 7 characters to exclude the year and surrounding parentheses
    movie_title = movie_title_with_year[:-7].strip() if len(movie_title_with_year) > 7 else movie_title_with_year

    # Add movie to movies dictionary
    movies_dict[movie_title] = movie_id

    # Add movie to category dictionary based on genre
    for genre, value in zip(genre_dict.keys(), map(int, parts[5:])):
        if value == 1:
            genre_dict[genre].append(movie_title)  # Add movie to the genre_dict as well

# Print or use the resulting dictionaries as needed
print(movies_dict)
print(genre_dict)

# Declare peers dictionary globally
peers = {user_id: {} for user_id in ratings['user_id'].unique()}

# Set of all users
U = set(ratings['user_id'].unique())


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


def get_movie_name_by_id(movie_id):
    for name, _id in movies_dict.items():
        if _id == movie_id:
            return name
    return None  # Return None if the movie ID is not found in the dictionary


# Function to provide explanations for why a certain recommendation may not be suitable for group members
def explain_why_not(input_string, why_not_question, rating_scores, recommendation_list, numPI, numP, peers,
                    neighborhood_size=5):
    explanations = {}  # Dictionary to store explanations for each group member

    for target_user_id in group_members:
        user_rated_movies = {} # Dictionary to store ratings for each group member
        explanation = set()  # Set to store unique explanations for each user

        similarity_scores = compute_user_similarity(target_user_id)
        user_rated_movies = ratings[ratings['user_id'] == user_id]['movie_id']
        id = movies_dict.get(input_string)

        if id is not None and id in user_rated_movies.values:
            explanation.add(f'User has already rated {input_string}')
        else:
            # Check if there is a tie with another item in the recommendation list.
            tied_items = [i0 for i0 in rating_scores.get(target_user_id, {}) if i0 != id and (rating_scores.get(target_user_id, {}).get(i0) == rating_scores.get(target_user_id, {}).get(id)).any() and recommendation_list.count(i0) <= 2 * neighborhood_size]
            if tied_items:
                explanation.add('Tied relevance scores')
            elif id not in user_rated_movies:
                explanation.add('User has not rated the item')
            # Check if the item appears between the kth and 2kth entry in the recommendation list.
            elif id in recommendation_list and recommendation_list.index(id) <= 2 * neighborhood_size:
                explanation.add('Position in list (neighborhood_size)')
            # Check if none of the users in the system have rated the item.
            elif id not in rating_scores:
                explanation.add('No peer ratings for the item')
            else:
                # Check the peers of the target user.
                if any(peers[target_user_id][peer][id] for peer in peers[target_user_id]):
                    for peer in peers[target_user_id]:
                        if peers[target_user_id][peer][id]:
                            explanation.add(
                                (f"Peer {peer}", f"Rating: {peers[target_user_id][peer][id]}",
                                 f"Similarity: {similarity_scores[peer]}"))
                    # Check if there are not enough most similar peers who have rated the item.
                    if len([1 for peer in peers[target_user_id] if peers[target_user_id][peer][id]]) < numPI:
                        explanation.add('Insufficient most similar peers')
                    # Check if there are not enough most similar peers who have rated the item among the top numP peers.
                    if len(peers[target_user_id]) < numP:
                        explanation.add('Insufficient overall similar peers')
                else:
                    # Check if none of the peers has rated the item.
                    for u0 in U:
                        if u0 != target_user_id and peers[target_user_id][u0][id]:
                            explanation.add((f"User {u0}", f"Rating: {peers[target_user_id][u0][id]}", '-'))
                    explanation.add('No peer ratings for the item')

        explanations[target_user_id] = explanation

    return explanations

def explain_group_why_not(input_string, why_not_question, peers_rating_scores, user_recommendations, numPI, numP, peers):
    collective_explanation = set()  # Set to store unique collective explanations for the group

    # Loop through each user in the group
    for target_user_id in group_members:
        user_rated_movies = ratings[ratings['user_id'] == target_user_id]['movie_id']

        # Get the top similar users and their similarity scores
        top_similar_users = get_top_similar_users(target_user_id)

        # Initialize a dictionary to store ratings for the current user
        user_ratings = {}

         # Loop through each similar user and their similarity score
        for similar_user_id, similarity_score in top_similar_users:
            # Add the similar user and their rating to the dictionary
            user_ratings[similar_user_id] = ratings[ratings['user_id'] == similar_user_id].set_index('movie_id')['rating']

        # Check if the category is already recommended to the user
        for movie, rating in user_recommendations[target_user_id]:
            movie_name = get_movie_name_by_id(movie)
            if movie_name in genre_dict[input_string]:
                collective_explanation.add(f"Category {input_string} is recommended to User {target_user_id}")
                continue  # Skip further checks for this user

        # Check if the category is already recommended to the user
        for movie in user_rated_movies:
            movie_name = get_movie_name_by_id(movie)
            if movie_name in genre_dict[input_string]:
                collective_explanation.add(f"User {target_user_id} has rated movie(s) from genre: {input_string}")
                continue  # Skip further checks for this user

        # Fetch movie names from genre_dict using movie iterator (which is an id number)
        movie_ids = [movies_dict.get(movie) for movie in genre_dict[input_string]]

        # Check if none of the users in the system have rated the items in the category.
        if not any(movie_id in movie_ids for movie_id in user_rated_movies):
            collective_explanation.add(f"No ratings for the items in Category {input_string} by User {target_user_id}")

        # Check the peers of the target user.
        if target_user_id in peers and any(peer in peers[target_user_id] for peer in peers[target_user_id]):
            for peer in peers[target_user_id]:
                if peer in peers[target_user_id] and peers[target_user_id][peer]:
                    for category in genre_dict[input_string]:
                        if category in peers[target_user_id][peer]:
                            explanation = (f"Peer {peer}: Rating: {peers[target_user_id][peer][category]}, "
                                           f"Similarity: {similarity_scores.get(peer, 'N/A')}")
                            collective_explanation.add(explanation)

        # Check if there are not enough most similar peers who have rated the items in the category.
        if len([1 for peer in peers[target_user_id] if any(
                category in peers[target_user_id][peer] for category in genre_dict[input_string])]) < numPI:
            collective_explanation.add(f"Insufficient most similar peers for Category {input_string} for User {target_user_id}")

        # Check if there are not enough most similar peers who have rated the items in the category among the top numP peers.
        if len(peers[target_user_id]) < numP:
            collective_explanation.add(f"Insufficient overall similar peers for Category {input_string} for User {target_user_id}")
            # Check if none of the peers has rated the items in the category.
            for u0 in U:
                if u0 != target_user_id and target_user_id in peers and u0 in peers[target_user_id] and any(
                        peers[target_user_id][u0].get(category) for category in genre_dict[input_string]):
                    explanation = f"User {u0}: Rating: {peers[target_user_id][u0][category]}"
                    collective_explanation.add(explanation)

        else:
            collective_explanation.add(f"No peer ratings for the items in Category {input_string} by User {target_user_id}")

    # Print explanations for the entire group
    print(f"\nWhy not {input_string}:")
    for reason in collective_explanation:
        print(f"- {reason}")

# C) Predict movie scores if rating doesn't exist
def predict_movie_score(user_id, movie_id, similar_users, floor=0.5):  # Ratings can't be less than 0.5
    user_ratings = ratings[ratings['user_id'] == user_id].set_index('movie_id')['rating']
    total_similarity = 0
    weighted_sum = 0
    predicted_score = 0

    if movie_id not in user_ratings.index:
        other_user_ratings = \
            ratings[(ratings['user_id'].isin(similar_users.keys())) & (ratings['movie_id'] == movie_id)].set_index(
                'user_id')[
                'rating']

        for user in user_ratings.index.intersection(other_user_ratings.index):
            similarity = similar_users[user]
            user_rating = user_ratings[user]
            other_user_rating = other_user_ratings[user]

            weighted_sum += similarity * other_user_rating
            total_similarity += abs(similarity)

        if total_similarity == 0:
            return None

        predicted_score = weighted_sum / total_similarity
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
def generate_group_of_users(num_members=3):  # number of members can be specified

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


# F.5 Calculate or fetch rating for a given movie for each member of the group
def calculate_user_ratings(user_ids, movie_id):
    # Initialize the dictionary to store ratings for each user in the group
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

    # Update the peers dictionary with user ratings for the given movie
    for user_id, rating in individual_ratings.items():
        peers[user_id][movie_id] = rating

    return individual_ratings


# F.6 Calculate top 10 recommendations with three different aggregation methods
def recommend_movies(movie_list, member_ids, aggregation_method, coefficient=0.2):
    # Dictionary to store aggregated scores for each movie
    aggregated_scores = {}

    # Store user similarities for all members and find existing ratings
    all_similarities = {}
    all_user_ratings = {}
    all_recommendation_lists = {}  # New dictionary to store recommendation lists

    for member_id in member_ids:
        all_similarities[member_id] = compute_user_similarity(member_id)
        all_user_ratings[member_id] = ratings[ratings['user_id'] == member_id].set_index('movie_id')['rating']
        all_recommendation_lists[member_id] = get_top_recommended_movies(member_id, all_similarities[member_id])

    # List to store extended recommendation lists for each member
    member_recommendations = []

    # Loop through each movie in the list
    for movie_id in movie_list:
        # Reset member_scores for the next movie
        member_scores = []
        movie_recommendations = []  # List to store recommendation lists for each member
        # Loop through each group member
        for member_id in member_ids:
            similar_users = all_similarities[member_id]
            user_ratings = all_user_ratings[member_id]

            # Check if the user has a rating for the movie
            if movie_id not in user_ratings.index:
                # If not, predict the rating using the predict_movie_score function
                prediction = predict_movie_score(member_id, movie_id, similar_users)

                # Add the movie_id to user_ratings with the predicted score
                user_ratings.at[
                    movie_id] = prediction if prediction is not None else 0  # Use 0 as a default if prediction is None

            # Add the score
            member_scores.append(user_ratings[movie_id])
            movie_recommendations.extend(all_recommendation_lists[member_id])  # Extend the recommendation list

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

            # Add the aggregated score to the dictionary
            aggregated_scores[movie_id] = aggregated_score

        # Append the recommendation list for the movie
        member_recommendations.append(movie_recommendations)

    # Sort and display the top 10 movies
    top_movies = sorted(aggregated_scores.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f'{aggregation_method} Aggregation: {top_movies}')

    # Return the top movies and extended recommendation lists
    return top_movies, member_recommendations

# PRINTS AND FUNCTION CALLS ###


# F.7 Generate group of 3 users
group_members = generate_group_of_users()

# F.12 Show top 10 movie recommendations for the group using three methods
print('\nTop 10 Movie Recommendations:')
top_movies, _ = recommend_movies(movie_list, group_members, 'Disagreement Aware')


print('\n####  ATOMIC CASE ####')
# Handle invalid movie input in the atomic case
while True:
    input_string = input("Why not ")  # movie name as input
    if input_string not in movies_dict:
        print("Invalid movie name. Please enter a valid movie.")
    else:
        break

# Create an empty dictionary to store explanations for each user in the group
explanations_dict = {}
peers_rating_scores = {}


# Loop through each user in the group
for user_id in group_members:
    # Get the top similar users and their similarity scores
    top_similar_users = get_top_similar_users(user_id)

    # Initialize a dictionary to store ratings for the current user
    user_ratings = {}

    # Loop through each similar user and their similarity score
    for similar_user_id, similarity_score in top_similar_users:

        # Add the similar user and their rating to the dictionary
        user_ratings[similar_user_id] = ratings[ratings['user_id'] == similar_user_id].set_index('movie_id')['rating']

    # Add the user's ratings to the overall dictionary
    peers_rating_scores[user_id] = user_ratings
    

    # Get the recommendation list for the current user
    user_recommendations = get_top_recommended_movies(user_id, compute_user_similarity(user_id))

    numPI = 5
    numP = 3
    # Call the explain_why_not function for the target movie ID
    explanations = explain_why_not(input_string, f"Why not {input_string}", peers_rating_scores, user_recommendations, numPI, numP, peers)

    # Store the explanations in the dictionary
    explanations_dict[user_id] = explanations[user_id]

# Print or use the explanations as needed
for user_id, explanation in explanations_dict.items():
    print(f"\nExplanations for User {user_id} and Movie Name {input_string}:")
    for reason in explanation:
        print(f"- {reason}")

print('\n####  GROUP CASE  ####')
# Handle invalid category input in the group case
while True:
    input_string = input("Why not ") # movie category as input
    if input_string not in genre_dict:
        print("Invalid category. Please enter a valid category.")
    else:
        break

# Create an empty dictionary to store explanations for each user in the group
cat_explanations_dict = {}
peers_rating_scores = {}


# Loop through each user in the group
for user_id in group_members:
    # Get the top similar users and their similarity scores
    top_similar_users = get_top_similar_users(user_id)

    # Initialize a dictionary to store ratings for the current user
    user_ratings = {}

    # Loop through each similar user and their similarity score
    for similar_user_id, similarity_score in top_similar_users:

        # Add the similar user and their rating to the dictionary
        user_ratings[similar_user_id] = ratings[ratings['user_id'] == similar_user_id].set_index('movie_id')['rating']

    # Add the user's ratings to the overall dictionary
    peers_rating_scores[user_id] = user_ratings
    
    # Get the recommendation list for the current user
    user_recommendations[user_id] = get_top_recommended_movies(user_id, compute_user_similarity(user_id))

numPI = 2
numP = 1
# Call the explain_group_why_not function for the target movie category
explain_group_why_not(input_string, f"Why not {input_string}", peers_rating_scores, user_recommendations, numPI, numP, peers)

print('\n####  POSITION ABSENTEEISM ####')
print('\nWhy not rank _ as first?')
# Handle invalid movie input in the atomic case
while True:
    input_string = input("Input movie: ")  # movie name as input
    if input_string not in movies_dict:
        print("Invalid movie name. Please enter a valid movie.")
    else:
        break

movie_id = movies_dict.get(input_string)
score_avg = {}

top_movie_ids = [t[0] for t in top_movies]

# Find if movie is already in top 10
if movie_id not in top_movie_ids:
    print(f"{input_string} doesn't have a high enough predicted group score to make the top 10")
# Explain why it isn't number 1
else:
    first_rank = max(top_movies, key=lambda k: k[1])
    key = first_rank[0]
    if movie_id != key:
        print(f"{input_string} is in the top 10 but it doesn't have a high enough predicted score to be ranked #1")
    else:
        print(f"{input_string} is already ranked number one")