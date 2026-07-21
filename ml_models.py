"""
ml_models.py

The extra ML layer on top of the base content-based engine
(recommendation.py just does genre cosine-similarity, which is closer
to a similarity heuristic than an actual trained model).

Three things live here:

1. KMeans clustering  -> groups movies into "mood" clusters purely
   from their genre vectors, so we can let people browse by vibe
   instead of only searching by title.

2. Item-based KNN collaborative filtering -> uses ratings.csv (which
   the original version downloaded but never actually used) to find
   movies that tend to get rated similarly by the same people. This
   gets blended with genre similarity for the "hybrid" recommend mode.

3. Linear Regression -> fits average rating against ln(num_ratings)
   to get a smoothed "expected rating" baseline, and uses the
   residual + rating count to rank a Trending Now list. Nothing
   fancy, just enough to show a regression model doing real work.

None of this needs to be perfect -- MovieLens ml-latest-small is a
pretty sparse dataset, so we fall back to content-based scores
wherever the ratings data isn't dense enough to trust.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors

N_CLUSTERS = 9
KNN_NEIGHBORS = 20

# rough, human-picked labels for what each genre combo usually feels
# like. clustering itself is unsupervised -- this just names the
# clusters after their most common genre so the UI doesn't have to
# show "Cluster 4".
_MOOD_NAMES = {
    "Action": "Adrenaline & Action",
    "Adventure": "Adventure & Quests",
    "Animation": "Animated & Family",
    "Children": "Animated & Family",
    "Comedy": "Comedy & Feel-Good",
    "Crime": "Crime & Thrillers",
    "Documentary": "Documentaries",
    "Drama": "Drama",
    "Fantasy": "Fantasy & Sci-Fi",
    "Sci-Fi": "Fantasy & Sci-Fi",
    "Horror": "Horror & Suspense",
    "Thriller": "Crime & Thrillers",
    "Romance": "Romance",
    "Musical": "Musicals",
    "War": "War & History",
    "Western": "Westerns",
    "Mystery": "Mystery",
}


# --- clustering -----------------------------------------------------

def build_movie_clusters(movies_df: pd.DataFrame, n_clusters: int = N_CLUSTERS):
    """
    Cluster movies by genre using KMeans. Returns the fitted movies_df
    with a new 'cluster' column, plus a dict mapping cluster id ->
    display name.

    We reuse the same CountVectorizer approach as the genre similarity
    matrix (bag-of-genres), just fed into KMeans instead of cosine
    similarity, so movies group into rough "mood" buckets.
    """
    genre_text = movies_df["genres"].str.replace("|", " ", regex=False)
    vectorizer = CountVectorizer()
    genre_vectors = vectorizer.fit_transform(genre_text)

    # small dataset, so plain KMeans is fine -- no need for MiniBatch here.
    k = min(n_clusters, genre_vectors.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(genre_vectors)

    clustered_df = movies_df.copy()
    clustered_df["cluster"] = labels

    cluster_names = _name_clusters(clustered_df, k)
    return clustered_df, cluster_names


def _name_clusters(clustered_df: pd.DataFrame, k: int) -> dict:
    """Pick a human-friendly name for each cluster from its most common genre."""
    names = {}
    used = set()
    for cluster_id in range(k):
        subset = clustered_df[clustered_df["cluster"] == cluster_id]
        genre_counts = {}
        for genres in subset["genres"]:
            for g in genres.split("|"):
                genre_counts[g] = genre_counts.get(g, 0) + 1

        # walk genres most-common-first and pick the first one that
        # maps to a mood name we haven't already used
        label = None
        for genre, _ in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True):
            candidate = _MOOD_NAMES.get(genre, genre)
            if candidate not in used:
                label = candidate
                break
        if label is None:
            label = f"Mixed Bag {cluster_id + 1}"

        names[cluster_id] = label
        used.add(label)
    return names


# --- collaborative filtering (item-based KNN) ------------------------

def build_collab_model(ratings_df: pd.DataFrame, movies_df: pd.DataFrame):
    """
    Build an item-based KNN model from the ratings matrix so we can
    find "people who rated this also rated..." style neighbors.

    Returns (knn_model, item_user_matrix, movieId_to_row) or
    (None, None, None) if there just isn't enough ratings data to
    build anything meaningful from (keeps the app from crashing on a
    tiny/empty ratings file).
    """
    if ratings_df is None or ratings_df.empty:
        return None, None, None

    # pivot to movies (rows) x users (cols), missing ratings -> 0.
    # this gets sparse fast, but ml-latest-small is small enough that
    # a dense pivot is still fine for a student project.
    pivot = ratings_df.pivot_table(index="movieId", columns="userId", values="rating").fillna(0)

    if pivot.shape[0] < KNN_NEIGHBORS:
        return None, None, None

    movie_id_to_row = {movie_id: i for i, movie_id in enumerate(pivot.index)}

    n_neighbors = min(KNN_NEIGHBORS + 1, pivot.shape[0])  # +1 because a movie is its own neighbor
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(pivot.values)

    return knn, pivot, movie_id_to_row


def get_collab_neighbors(movie_id, knn_model, pivot, movie_id_to_row) -> dict:
    """
    Return {movieId: similarity_score} for movies whose rating
    patterns are closest to the given movie, according to the KNN
    model. Empty dict if the model isn't available or the movie
    wasn't rated enough to be in it.
    """
    if knn_model is None or movie_id not in movie_id_to_row:
        return {}

    row_idx = movie_id_to_row[movie_id]
    distances, indices = knn_model.kneighbors([pivot.values[row_idx]])

    row_to_movie_id = {v: k for k, v in movie_id_to_row.items()}
    scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        neighbor_id = row_to_movie_id[idx]
        if neighbor_id == movie_id:
            continue
        # cosine distance -> similarity
        scores[neighbor_id] = max(0.0, 1 - dist)
    return scores


# --- trending score via linear regression -----------------------------

def compute_trending_scores(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fit a simple Linear Regression of avg_rating ~ ln(num_ratings) to
    get a "how well-rated is this movie for how much attention it's
    gotten" baseline, then rank movies by how far above that baseline
    they sit, weighted by how many ratings back them up.

    This is basically a regression-flavoured take on the classic
    "weighted rating" trending formula (same idea IMDb's top-250 uses,
    just with the baseline learned instead of hand-picked).

    Returns movies_df with two new columns: num_ratings, trending_score
    -- sorted descending by trending_score. If there's no ratings data
    at all, trending_score just falls back to 0 for every row.
    """
    if ratings_df is None or ratings_df.empty:
        out = movies_df.copy()
        out["num_ratings"] = 0
        out["trending_score"] = 0.0
        return out

    agg = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"]).reset_index()
    agg.columns = ["movieId", "avg_rating", "num_ratings"]

    merged = movies_df.merge(agg, on="movieId", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0)
    merged["num_ratings"] = merged["num_ratings"].fillna(0)

    rated = merged[merged["num_ratings"] > 0]
    if len(rated) < 10:
        merged["trending_score"] = merged["avg_rating"] * np.log1p(merged["num_ratings"])
        return merged.sort_values("trending_score", ascending=False).reset_index(drop=True)

    X = np.log1p(rated["num_ratings"]).values.reshape(-1, 1)
    y = rated["avg_rating"].values
    reg = LinearRegression()
    reg.fit(X, y)

    # predict the "expected" rating for every movie given its rating count,
    # then see how far each movie beats (or misses) that expectation.
    merged["expected_rating"] = reg.predict(np.log1p(merged["num_ratings"]).values.reshape(-1, 1))
    merged["rating_lift"] = merged["avg_rating"] - merged["expected_rating"]

    # blend the "beats expectations" signal with raw popularity so a
    # movie with 3 perfect ratings doesn't outrank one with hundreds
    # of consistently good ones.
    merged["trending_score"] = merged["rating_lift"] * np.log1p(merged["num_ratings"])
    merged.loc[merged["num_ratings"] == 0, "trending_score"] = -999  # never rated -> never trending

    return merged.sort_values("trending_score", ascending=False).reset_index(drop=True)
