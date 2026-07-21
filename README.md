# 🎬 Moviefy

A movie recommendation system built with **Streamlit**. Started as a
pure content-based recommender (genre similarity via CountVectorizer +
Cosine Similarity) on the **MovieLens ml-latest-small** dataset, and
now also uses the ratings data for a real ML layer on top:

- **KMeans** clusters movies into mood groups for browsing
- **Item-based KNN** (`sklearn.neighbors.NearestNeighbors`) does
  collaborative filtering off the ratings matrix, blended with genre
  similarity for a "hybrid" recommend mode
- **Linear Regression** fits average rating against ratings-volume to
  power a Trending Now ranking (rewards movies that beat their
  "expected" rating for their popularity, not just raw star count)

Posters, ratings, plot, and cast are pulled live from the **OMDb API**.

Built as a B.Tech student project — kept modular and readable rather
than over-engineered.

---

## Features

- Content-based recommendations using genre similarity
- Hybrid mode that blends in a KNN collaborative-filtering model
  trained on user ratings (falls back to content-only automatically if
  there isn't enough ratings data to train on)
- "Explore by mood" — KMeans-clustered browsing when you don't have a
  specific title in mind
- "Trending Now" strip ranked by a Linear Regression popularity model
- Choose Top 5 or Top 10 recommendations, narrow by genre, or hit
  "Surprise me"
- Movie posters, IMDb ratings, plot, cast, and director via OMDb API
- Cached pickle files (`movies.pkl`, `ratings.pkl`, `similarity.pkl`)
  for fast reloads
- Clean dark-themed, responsive UI
- Graceful error handling — never crashes on missing data or API errors

---

## Project Structure

```text
Moviefy/
│
├── app.py                 # Streamlit UI and main entry point
├── preprocess.py          # Dataset download + cleaning (movies + ratings)
├── recommendation.py      # Content-based engine + hybrid blending
├── ml_models.py           # KMeans clustering, KNN collab filtering, trending regression
├── requirements.txt
├── movies.pkl              # generated on first run
├── ratings.pkl              # generated on first run
├── similarity.pkl          # generated on first run
│
├── dataset/                # MovieLens CSVs (auto-downloaded)
├── css/
│   └── style.css
├── assets/
│   └── logo.png
└── .streamlit/
    ├── config.toml
    └── secrets.toml.example
```

---

## Setup (Local)

1. **Clone the project and install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OMDb API key** (get a free key at
   https://www.omdbapi.com/apikey.aspx):

   ```bash
   export OMDB_API_KEY=your_key_here      # macOS/Linux
   set OMDB_API_KEY=your_key_here         # Windows (cmd)
   ```

   Or copy `.env.example` to `.env` and load it with a tool like
   `python-dotenv`, or use Streamlit secrets (see below).

3. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   On first run, `app.py` automatically downloads the MovieLens
   dataset, cleans it, builds the similarity matrix, and caches both
   as `movies.pkl` and `similarity.pkl`. Subsequent runs load the
   cached files instantly.

---

## Deploying to Streamlit Community Cloud

1. Push this project to a GitHub repository (the `.gitignore`
   already excludes generated pickle files and datasets — they'll
   be rebuilt automatically on first load).
2. Go to [share.streamlit.io](https://share.streamlit.io) and create
   a new app pointing to `app.py`.
3. In the app's **Settings → Secrets**, paste:

   ```toml
   OMDB_API_KEY = "your_omdb_api_key_here"
   ```

   (see `.streamlit/secrets.toml.example` for reference)

4. Deploy. The app installs `requirements.txt`, downloads the
   dataset, and builds its cache automatically.

---

## How It Works

1. **`preprocess.py`** downloads `ml-latest-small.zip`, extracts
   `movies.csv` and `ratings.csv`, cleans both (dupes, missing values,
   genre-less entries), and saves them as `movies.pkl` / `ratings.pkl`.

2. **`recommendation.py`** builds the genre similarity matrix
   (CountVectorizer + cosine similarity) and exposes
   `get_recommendations()`. It also has `get_hybrid_recommendations()`,
   which blends that with the collaborative-filtering scores from
   `ml_models.py`.

3. **`ml_models.py`** is the ML layer that uses the ratings data:
   - `build_movie_clusters()` — KMeans over genre vectors, used for
     the "Explore by mood" browsing section
   - `build_collab_model()` / `get_collab_neighbors()` — item-based
     KNN (cosine distance) over the user-movie ratings matrix
   - `compute_trending_scores()` — Linear Regression of average
     rating against ln(rating count), used to rank the Trending Now
     strip by how much a movie over/under-performs expectations for
     its popularity

4. **`app.py`** ties it together: a Streamlit UI lets the user pick a
   movie, a recommendation count, and whether to factor in community
   ratings, then displays results as cards enriched with live OMDb
   data. When there's no active search, it shows the Trending Now and
   Explore by Mood sections instead of a blank state.

---

## Notes

- The OMDb API key is never hardcoded — it's read from
  `st.secrets` (Streamlit Cloud) or the `OMDB_API_KEY` environment
  variable, with a fallback demo key so the app still runs out of
  the box for local testing.
- If OMDb is unreachable or a movie isn't found, the app shows a
  placeholder poster and "N/A" for missing fields instead of
  crashing.
- The hybrid/collaborative-filtering features degrade gracefully: if
  `ratings.pkl` is missing or too sparse to train a useful KNN model,
  the app just runs in pure content-based mode and disables the
  "Factor in community ratings" toggle.
