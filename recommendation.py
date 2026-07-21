"""
recommendation.py

The content-based half of the recommender: builds a genre similarity
matrix with CountVectorizer + cosine similarity, and exposes
get_recommendations() for the Streamlit app.

(The hybrid version that blends in collaborative filtering lives at
the bottom of this file, get_hybrid_recommendations() -- it leans on
ml_models.py for the KNN piece.)

Run this file directly to (re)generate similarity.pkl:
    python recommendation.py
"""

import os

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocess import run_preprocessing, MOVIES_PKL_PATH

SIMILARITY_PKL_PATH = "similarity.pkl"


# ---------------------------------------------------------------------------
# Loading / building the movies dataframe
# ---------------------------------------------------------------------------
def load_movies() -> pd.DataFrame:
    """
    Load the cleaned movies dataframe from movies.pkl.
    If it doesn't exist yet, run the preprocessing pipeline first.
    """
    if not os.path.exists(MOVIES_PKL_PATH):
        return run_preprocessing()
    return pd.read_pickle(MOVIES_PKL_PATH)


# ---------------------------------------------------------------------------
# Building the similarity matrix
# ---------------------------------------------------------------------------
def build_similarity_matrix(movies_df: pd.DataFrame):
    """
    Convert the genres column into count vectors and compute the
    cosine similarity between every pair of movies.

    Genres in the dataset look like "Adventure|Animation|Comedy".
    We replace the '|' separator with a space so CountVectorizer
    treats each genre as its own token/word.
    """
    genre_text = movies_df["genres"].str.replace("|", " ", regex=False)

    vectorizer = CountVectorizer()
    genre_vectors = vectorizer.fit_transform(genre_text)

    similarity_matrix = cosine_similarity(genre_vectors)
    return similarity_matrix


def save_similarity_matrix(similarity_matrix) -> None:
    """Save the similarity matrix as a pickle file for fast reuse."""
    pd.to_pickle(similarity_matrix, SIMILARITY_PKL_PATH)
    print(f"Similarity matrix saved to {SIMILARITY_PKL_PATH}.")


def load_or_build_similarity_matrix(movies_df: pd.DataFrame):
    """
    Load the similarity matrix from similarity.pkl if it exists,
    otherwise build it fresh and cache it to disk.
    """
    if os.path.exists(SIMILARITY_PKL_PATH):
        return pd.read_pickle(SIMILARITY_PKL_PATH)

    similarity_matrix = build_similarity_matrix(movies_df)
    save_similarity_matrix(similarity_matrix)
    return similarity_matrix


# ---------------------------------------------------------------------------
# Public recommendation function
# ---------------------------------------------------------------------------
def get_recommendations(movie_title: str, n: int = 5, movies_df=None, similarity_matrix=None):
    """
    Return the top-n movies most similar to the given movie title,
    based on genre similarity.

    Parameters
    ----------
    movie_title : str
        The exact title of the movie to base recommendations on
        (must match a title in movies.pkl).
    n : int
        Number of recommendations to return. Expected to be 5 or 10.
    movies_df : pd.DataFrame, optional
        Preloaded movies dataframe. If not provided, it is loaded
        from disk (slower -- prefer passing it in from the app).
    similarity_matrix : np.ndarray, optional
        Preloaded similarity matrix. If not provided, it is loaded
        or built from disk.

    Returns
    -------
    pd.DataFrame
        A dataframe of the top-n recommended movies with an added
        'similarity_score' column, sorted by similarity descending.
        Returns an empty dataframe if the title is not found.
    """
    if movies_df is None:
        movies_df = load_movies()
    if similarity_matrix is None:
        similarity_matrix = load_or_build_similarity_matrix(movies_df)

    # Find the index of the requested movie.
    matches = movies_df.index[movies_df["title"] == movie_title].tolist()
    if not matches:
        return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

    movie_index = matches[0]

    # Get similarity scores for this movie against all others.
    scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort by similarity score, descending. Skip index 0 result
    # since that will always be the movie itself (score == 1.0).
    scores = sorted(scores, key=lambda item: item[1], reverse=True)
    scores = [item for item in scores if item[0] != movie_index][:n]

    recommended_indices = [item[0] for item in scores]
    similarity_scores = [item[1] for item in scores]

    recommendations = movies_df.iloc[recommended_indices].copy()
    recommendations["similarity_score"] = similarity_scores
    recommendations = recommendations.sort_values("similarity_score", ascending=False)
    recommendations = recommendations.reset_index(drop=True)

    return recommendations


if __name__ == "__main__":
    # Quick manual test when running this file directly.
    movies = load_movies()
    sim_matrix = load_or_build_similarity_matrix(movies)

    sample_title = movies["title"].iloc[0]
    print(f"Sample recommendations for: {sample_title}\n")
    print(get_recommendations(sample_title, n=5, movies_df=movies, similarity_matrix=sim_matrix))


# ---------------------------------------------------------------------------
# Hybrid recommendations (content-based + collaborative filtering)
# ---------------------------------------------------------------------------
def get_hybrid_recommendations(
    movie_title: str,
    n: int = 5,
    movies_df=None,
    similarity_matrix=None,
    knn_model=None,
    ratings_pivot=None,
    movie_id_to_row=None,
    collab_weight: float = 0.45,
):
    """
    Same idea as get_recommendations(), but blends in the item-based
    KNN collaborative filtering scores from ml_models.py where
    they're available.

    final_score = (1 - collab_weight) * content_similarity
                  + collab_weight * collaborative_similarity

    If the requested movie has no (or too few) ratings for the KNN
    model to use, this just quietly falls back to pure content-based
    scoring -- no crash, no empty result.
    """
    from ml_models import get_collab_neighbors  # local import avoids a circular dependency

    if movies_df is None:
        movies_df = load_movies()
    if similarity_matrix is None:
        similarity_matrix = load_or_build_similarity_matrix(movies_df)

    matches = movies_df.index[movies_df["title"] == movie_title].tolist()
    if not matches:
        return pd.DataFrame(columns=["movieId", "title", "genres", "similarity_score"])

    movie_index = matches[0]
    movie_id = movies_df.iloc[movie_index]["movieId"]

    content_scores = list(enumerate(similarity_matrix[movie_index]))
    content_scores = {idx: score for idx, score in content_scores if idx != movie_index}

    collab_scores_by_id = {}
    if knn_model is not None:
        collab_scores_by_id = get_collab_neighbors(movie_id, knn_model, ratings_pivot, movie_id_to_row)

    # collab_scores_by_id is keyed by movieId, content_scores by row index,
    # so look candidate movieId up per row when blending the two.
    combined = {}
    for idx, content_score in content_scores.items():
        candidate_id = movies_df.iloc[idx]["movieId"]
        collab_score = collab_scores_by_id.get(candidate_id, 0.0)
        if collab_scores_by_id:
            combined[idx] = (1 - collab_weight) * content_score + collab_weight * collab_score
        else:
            combined[idx] = content_score  # no collab data for this movie, stick with content-only

    ranked = sorted(combined.items(), key=lambda item: item[1], reverse=True)[:n]
    recommended_indices = [idx for idx, _ in ranked]
    scores = [score for _, score in ranked]

    recommendations = movies_df.iloc[recommended_indices].copy()
    recommendations["similarity_score"] = scores
    recommendations = recommendations.reset_index(drop=True)
    return recommendations
