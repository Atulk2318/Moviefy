"""
preprocess.py

Grabs the MovieLens ml-latest-small dataset (if we don't have it yet),
cleans movies.csv with Pandas, and also preps ratings.csv since the
collaborative-filtering part of the recommender needs it later.

Just run `python preprocess.py` directly to regenerate the pickle
files from scratch (e.g. after deleting them or tweaking the cleaning
logic).
"""

import os
import zipfile
import urllib.request

import pandas as pd

DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATASET_DIR = "dataset"
ZIP_PATH = os.path.join(DATASET_DIR, "ml-latest-small.zip")
EXTRACTED_FOLDER_NAME = "ml-latest-small"  # folder name inside the zip

MOVIES_CSV_PATH = os.path.join(DATASET_DIR, "movies.csv")
RATINGS_CSV_PATH = os.path.join(DATASET_DIR, "ratings.csv")

MOVIES_PKL_PATH = "movies.pkl"
RATINGS_PKL_PATH = "ratings.pkl"

# a movie needs at least this many ratings before the collab-filtering
# model bothers using it -- below this it's too noisy to trust.
MIN_RATINGS_THRESHOLD = 5


def download_dataset() -> None:
    """
    Download and extract the MovieLens ml-latest-small dataset if
    movies.csv and ratings.csv are not already present in dataset/.
    """
    if os.path.exists(MOVIES_CSV_PATH) and os.path.exists(RATINGS_CSV_PATH):
        print("Dataset already present. Skipping download.")
        return

    os.makedirs(DATASET_DIR, exist_ok=True)

    print("Downloading MovieLens dataset...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("Download complete. Extracting...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATASET_DIR)

    # The zip extracts into a subfolder named "ml-latest-small".
    # Move the two CSV files we need up into dataset/ directly.
    extracted_folder = os.path.join(DATASET_DIR, EXTRACTED_FOLDER_NAME)
    for filename in ("movies.csv", "ratings.csv"):
        src = os.path.join(extracted_folder, filename)
        dst = os.path.join(DATASET_DIR, filename)
        if os.path.exists(src):
            os.replace(src, dst)

    # Clean up the zip file and now-empty extracted folder.
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    if os.path.isdir(extracted_folder):
        try:
            os.rmdir(extracted_folder)
        except OSError:
            pass  # Folder not empty (extra files) -- safe to leave as is.

    print("Dataset ready in dataset/ folder.")


# --- cleaning -----------------------------------------------------

def clean_movies_data() -> pd.DataFrame:
    """
    Load movies.csv and tidy it up: drop dupes, drop rows missing a
    title/genre, keep only the columns we care about, and get rid of
    movies with "(no genres listed)" since CountVectorizer needs real
    genre text to do anything useful with them.
    """
    movies_df = pd.read_csv(MOVIES_CSV_PATH)
    movies_df = movies_df[["movieId", "title", "genres"]]
    movies_df = movies_df.drop_duplicates()
    movies_df = movies_df.dropna(subset=["title", "genres"])
    movies_df = movies_df[movies_df["genres"] != "(no genres listed)"]
    movies_df = movies_df.reset_index(drop=True)
    return movies_df


def clean_ratings_data(valid_movie_ids=None) -> pd.DataFrame:
    """
    Load ratings.csv and clean it up for the collaborative-filtering
    side of things (see ml_models.py). We don't need the timestamp
    column for anything, and it's worth dropping ratings for movies
    that got filtered out of movies_df above so the two stay in sync.
    """
    ratings_df = pd.read_csv(RATINGS_CSV_PATH)
    ratings_df = ratings_df[["userId", "movieId", "rating"]]
    ratings_df = ratings_df.dropna()
    ratings_df = ratings_df.drop_duplicates(subset=["userId", "movieId"])

    if valid_movie_ids is not None:
        ratings_df = ratings_df[ratings_df["movieId"].isin(valid_movie_ids)]

    ratings_df = ratings_df.reset_index(drop=True)
    return ratings_df


def save_cleaned_data(movies_df: pd.DataFrame, ratings_df: pd.DataFrame = None) -> None:
    """Pickle the cleaned dataframes so we don't have to redo this every run."""
    movies_df.to_pickle(MOVIES_PKL_PATH)
    print(f"Cleaned dataset saved to {MOVIES_PKL_PATH} ({len(movies_df)} movies).")
    if ratings_df is not None:
        ratings_df.to_pickle(RATINGS_PKL_PATH)
        print(f"Cleaned ratings saved to {RATINGS_PKL_PATH} ({len(ratings_df)} ratings).")


def run_preprocessing() -> pd.DataFrame:
    """Full pipeline: download if needed, clean movies + ratings, save both."""
    download_dataset()
    movies_df = clean_movies_data()
    ratings_df = clean_ratings_data(valid_movie_ids=set(movies_df["movieId"]))
    save_cleaned_data(movies_df, ratings_df)
    return movies_df


if __name__ == "__main__":
    run_preprocessing()
