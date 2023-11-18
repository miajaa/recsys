# F.13 Disagreements-Aware Aggregation Method
def group_recommendation_disagreements(user_ids, movie_id, coefficient=0.2):    # Disagreement is weighted using coefficient
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

    # Handle the case where there is only one user in the group or there's no disagreements
    if len(individual_ratings) == 1 or disagreements == 0:
        return round(list(individual_ratings.values())[0], 3)

    # Calculate the disagreements-aware aggregation
    weighted_avg_disagreements = np.mean(list(individual_ratings.values())) + coefficient * disagreements

    return round(weighted_avg_disagreements, 3)