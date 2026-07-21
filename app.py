"""
app.py

Moviefy - content-based + collaborative movie recommender built with
Streamlit. "Silver Screen" theme: dark, editorial, cinema-poster feel.

Run with:
    streamlit run app.py
"""

import os
import random
import re

import requests
import streamlit as st

from preprocess import run_preprocessing, MOVIES_PKL_PATH, RATINGS_PKL_PATH
from recommendation import (
    load_movies,
    load_or_build_similarity_matrix,
    get_recommendations,
    get_hybrid_recommendations,
)
from ml_models import build_movie_clusters, build_collab_model, compute_trending_scores

# Constants
OMDB_API_URL = "https://www.omdbapi.com/"
PLACEHOLDER_POSTER = (
    "https://placehold.co/400x600/101010/c9a25c?text=Moviefy&font=raleway"
)


def _get_omdb_api_key() -> str:
    try:
        if "OMDB_API_KEY" in st.secrets:
            return st.secrets["OMDB_API_KEY"]
    except Exception:
        pass
    return os.environ.get("OMDB_API_KEY", "81038dd6")


OMDB_API_KEY = _get_omdb_api_key()

# ─────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Moviefy — Find your next favorite film",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
# CSS loader
# ─────────────────────────────────────────────
def load_css(path: str) -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Cached data
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_movies_data():
    if not os.path.exists(MOVIES_PKL_PATH):
        return run_preprocessing()
    return load_movies()


@st.cache_resource(show_spinner=False)
def get_similarity_data(_movies_df):
    return load_or_build_similarity_matrix(_movies_df)


@st.cache_data(show_spinner=False)
def get_ratings_data():
    # ratings.pkl only exists once preprocessing has run at least once --
    # if it's missing (fresh clone, first run) just skip collab features
    # instead of blowing up the whole app.
    if not os.path.exists(RATINGS_PKL_PATH):
        return None
    try:
        import pandas as pd
        return pd.read_pickle(RATINGS_PKL_PATH)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_cluster_data(movies_df):
    return build_movie_clusters(movies_df)


@st.cache_resource(show_spinner=False)
def get_collab_model(_ratings_df, _movies_df):
    return build_collab_model(_ratings_df, _movies_df)


@st.cache_data(show_spinner=False)
def get_trending_data(_ratings_df, movies_df):
    return compute_trending_scores(_ratings_df, movies_df)


def clean_title_for_search(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def extract_year(title: str) -> str:
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else "N/A"


@st.cache_data(show_spinner=False)
def get_all_genres(movies_df) -> list:
    genre_set = set()
    for gs in movies_df["genres"]:
        genre_set.update(gs.split("|"))
    return sorted(g for g in genre_set if g != "(no genres listed)")


# ─────────────────────────────────────────────
# OMDb
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3_600)
def fetch_movie_details(title: str) -> dict:
    fallback = {
        "poster": PLACEHOLDER_POSTER,
        "imdb_rating": "N/A",
        "genre": "N/A",
        "year": extract_year(title),
        "plot": "No plot description available.",
        "director": "N/A",
        "actors": "N/A",
        "runtime": "N/A",
        "found": False,
    }
    search_title = clean_title_for_search(title)
    search_year = extract_year(title)
    params = {"t": search_title, "apikey": OMDB_API_KEY}
    if search_year != "N/A":
        params["y"] = search_year
    try:
        r = requests.get(OMDB_API_URL, params=params, timeout=6)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return fallback
    if data.get("Response") != "True":
        return fallback
    poster = data.get("Poster")
    if not poster or poster == "N/A":
        poster = PLACEHOLDER_POSTER
    return {
        "poster": poster,
        "imdb_rating": data.get("imdbRating", "N/A"),
        "genre": data.get("Genre", "N/A"),
        "year": data.get("Year", search_year),
        "plot": data.get("Plot", "No plot description available."),
        "director": data.get("Director", "N/A"),
        "actors": data.get("Actors", "N/A"),
        "runtime": data.get("Runtime", "N/A"),
        "found": True,
    }


# ─────────────────────────────────────────────
# UI — Hero
# ─────────────────────────────────────────────
def render_hero(movies_df) -> None:
    total_movies = len(movies_df)
    total_genres = len(get_all_genres(movies_df))

    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-eyebrow">Moviefy &mdash; Content-Based Discovery</div>
          <h1 class="hero-title">Find your next<br><em>favorite film.</em></h1>
          <p class="hero-sub">
            Pick a movie you already love. Moviefy reads its genre DNA and
            surfaces the closest matches from a catalog of thousands &mdash;
            no ratings history, no accounts, just taste.
          </p>
          <div class="hero-stats">
            <div class="hero-stat">
              <span class="num">{total_movies:,}</span>
              <span class="label">Titles indexed</span>
            </div>
            <div class="hero-stat">
              <span class="num">{total_genres}</span>
              <span class="label">Genre tags</span>
            </div>
            <div class="hero-stat">
              <span class="num">&lt;1s</span>
              <span class="label">Match time</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# UI — Cinema Card
# ─────────────────────────────────────────────
def render_cinema_card(
    movie_row, details: dict, match_label: str | None = None
) -> None:
    """Render a premium cinema-style movie card."""
    genres_raw = details["genre"] if details["genre"] != "N/A" else movie_row["genres"]
    genres_display = " &middot; ".join(
        g.strip() for g in genres_raw.replace("|", ",").split(",")[:3]
    )
    year = details["year"] if details["year"] != "N/A" else extract_year(movie_row["title"])
    rating = details["imdb_rating"]
    director = details["director"]
    actors = details["actors"]
    title = movie_row["title"]
    # Strip year suffix from display title
    display_title = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()

    rating_html = (
        f'<div class="cinema-rating">&#9733; {rating}</div>'
        if rating != "N/A"
        else ""
    )
    match_html = (
        f'<div class="cinema-match">{match_label}</div>' if match_label else ""
    )
    hover_html = ""
    if director != "N/A" or actors != "N/A":
        dir_line = (
            f'<div class="cinema-hover-director">Dir. {director}</div>'
            if director != "N/A"
            else ""
        )
        cast_line = (
            f'<div class="cinema-hover-actors">{actors}</div>'
            if actors != "N/A"
            else ""
        )
        hover_html = f'<div class="cinema-hover-info">{dir_line}{cast_line}</div>'

    eyebrow = f"ADMIT ONE &middot; {year}" if year != "N/A" else "ADMIT ONE"

    st.markdown(
        f"""
        <div class="cinema-card">
          <div class="cinema-card-media">
            <img src="{details['poster']}"
                 alt="{display_title}"
                 onerror="this.src='{PLACEHOLDER_POSTER}'" />
            {rating_html}
            {match_html}
            {hover_html}
          </div>
          <div class="cinema-card-body">
            <div class="cinema-card-eyebrow">{eyebrow}</div>
            <div class="cinema-card-title">{display_title}</div>
            <div class="cinema-card-genres">{genres_display}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Details"):
        plot = details.get("plot", "N/A")
        runtime = details.get("runtime", "N/A")
        st.markdown(
            f"""
            <div class="detail-panel">
              <div class="detail-plot">{plot}</div>
              <div class="detail-panel-grid">
                <div class="detail-row">
                  <strong>Director</strong>
                  {director}
                </div>
                <div class="detail-row">
                  <strong>Cast</strong>
                  {actors}
                </div>
                <div class="detail-row">
                  <strong>Runtime</strong>
                  {runtime}
                </div>
                <div class="detail-row">
                  <strong>IMDb Rating</strong>
                  {rating}
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
# UI — Cards Grid
# ─────────────────────────────────────────────
def render_cards_grid(recommendations, cards_per_row: int = 5, label_formatter=None) -> None:
    """
    label_formatter(row, position) -> str | None lets callers swap out
    the default "NN% match" badge for something else (e.g. a trending
    rank) without duplicating this whole grid-layout function.
    """
    rows_needed = -(-len(recommendations) // cards_per_row)
    for row_i in range(rows_needed):
        cols = st.columns(cards_per_row)
        start = row_i * cards_per_row
        slice_ = recommendations.iloc[start : start + cards_per_row]
        for col, (pos, (_, movie_row)) in zip(cols, enumerate(slice_.iterrows(), start=start)):
            with col:
                details = fetch_movie_details(movie_row["title"])
                if label_formatter:
                    match_label = label_formatter(movie_row, pos)
                elif (
                    "similarity_score" in movie_row
                    and movie_row["similarity_score"] is not None
                ):
                    pct = round(movie_row["similarity_score"] * 100)
                    match_label = f"{pct}% match"
                else:
                    match_label = None
                render_cinema_card(movie_row, details, match_label)


# ─────────────────────────────────────────────
# UI — Section title helper
# ─────────────────────────────────────────────
def section_title(text: str, count: int | None = None) -> None:
    count_html = (
        f'<span class="section-count">&nbsp;{count} films</span>' if count else ""
    )
    st.markdown(
        f'<div class="section-head"><h2 class="section-title">{text}</h2>{count_html}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# UI — Trending strip (Linear Regression ranked)
# ─────────────────────────────────────────────
def render_trending_strip(trending_df, movies_df) -> None:
    top = trending_df[trending_df["num_ratings"] > 0].head(6)
    if top.empty:
        return

    section_title("Trending <em>now</em>")
    st.caption("Ranked by a Linear Regression model of rating vs. popularity — not just raw averages")
    render_cards_grid(
        top,
        cards_per_row=6,
        label_formatter=lambda row, pos: f"#{pos + 1} trending",
    )
    st.markdown("<hr/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# UI — Explore by mood (KMeans clusters)
# ─────────────────────────────────────────────
def render_mood_explorer(clustered_df, cluster_names: dict) -> None:
    section_title("Explore by <em>mood</em>")
    st.caption("Movies grouped by genre DNA using KMeans clustering — pick a vibe")

    cluster_ids = sorted(cluster_names.keys())
    cols = st.columns(len(cluster_ids))
    for col, cid in zip(cols, cluster_ids):
        with col:
            if st.button(cluster_names[cid], key=f"mood_{cid}", use_container_width=True):
                st.session_state["selected_mood"] = cid

    selected = st.session_state.get("selected_mood")
    if selected is not None:
        subset = clustered_df[clustered_df["cluster"] == selected]
        sample_size = min(6, len(subset))
        sample = subset.sample(sample_size, random_state=None) if sample_size else subset
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.caption(f"{cluster_names[selected]} · {len(subset)} titles in the catalog")
        render_cards_grid(sample.reset_index(drop=True), cards_per_row=6, label_formatter=lambda row, pos: None)
        if st.button("Clear mood filter"):
            st.session_state["selected_mood"] = None
            st.rerun()

    st.markdown("<hr/>", unsafe_allow_html=True)


def main() -> None:
    load_css("css/style.css")

    with st.spinner("Loading the catalog…"):
        movies_df = get_movies_data()
        similarity_matrix = get_similarity_data(movies_df)
        ratings_df = get_ratings_data()
        clustered_df, cluster_names = get_cluster_data(movies_df)
        knn_model, ratings_pivot, movie_id_to_row = get_collab_model(ratings_df, movies_df)
        trending_df = get_trending_data(ratings_df, movies_df)

    render_hero(movies_df)

    # ── Controls ─────────────────────────────
    search_col, genre_col = st.columns([5, 3])

    with search_col:
        movie_title = st.selectbox(
            "Start with a movie you like",
            options=movies_df["title"].sort_values().tolist(),
            index=None,
            placeholder="Search titles…",
        )

    all_genres = get_all_genres(movies_df)
    with genre_col:
        genre_filter = st.multiselect(
            "Narrow by genre",
            options=all_genres,
            placeholder="Any genre",
        )

    mode_col, count_col, action_col, surprise_col = st.columns([2, 1, 1, 1])
    with mode_col:
        use_hybrid = st.toggle(
            "Factor in community ratings",
            value=knn_model is not None,
            disabled=knn_model is None,
            help=(
                "Blends genre similarity with an item-based KNN model trained on "
                "MovieLens ratings. Turned off automatically if there isn't enough "
                "ratings data to train on."
                if knn_model is not None
                else "Not enough ratings data loaded to train the collaborative model yet."
            ),
        )
    with count_col:
        recommendation_count = st.selectbox("How many?", options=[5, 10], index=0)
    with action_col:
        st.write("")
        recommend_clicked = st.button(
            "Get recommendations", type="primary", use_container_width=True
        )
    with surprise_col:
        st.write("")
        surprise_clicked = st.button(
            "Surprise me", type="secondary", use_container_width=True
        )

    st.markdown("<hr/>", unsafe_allow_html=True)

    def _recommend(title):
        if use_hybrid and knn_model is not None:
            return get_hybrid_recommendations(
                title,
                n=recommendation_count,
                movies_df=movies_df,
                similarity_matrix=similarity_matrix,
                knn_model=knn_model,
                ratings_pivot=ratings_pivot,
                movie_id_to_row=movie_id_to_row,
            )
        return get_recommendations(
            title, n=recommendation_count, movies_df=movies_df, similarity_matrix=similarity_matrix
        )

    # ── Surprise Me ──────────────────────────
    if surprise_clicked:
        pool = movies_df
        if genre_filter:
            mask = movies_df["genres"].apply(
                lambda g: any(genre in g.split("|") for genre in genre_filter)
            )
            pool = movies_df[mask]
        if pool.empty:
            st.warning("No movies match that genre filter. Try clearing it.")
            return

        random_title = random.choice(pool["title"].tolist())
        display = re.sub(r"\s*\(\d{4}\)\s*$", "", random_title).strip()

        with st.spinner(f'Pulling a ticket for "{display}"…'):
            recs = _recommend(random_title)

        section_title(
            f"Tonight's pick: <em>{display}</em>"
        )
        st.caption("Random pick from the catalog · closest matches below")

        # Show the picked movie hero card
        hero_col, _ = st.columns([1, 4])
        with hero_col:
            picked_row = movies_df[movies_df["title"] == random_title].iloc[0]
            details = fetch_movie_details(random_title)
            render_cinema_card(picked_row, details)

        if not recs.empty:
            st.markdown("<hr/>", unsafe_allow_html=True)
            section_title("Similar <em>titles</em>", count=len(recs))
            render_cards_grid(recs, cards_per_row=5)
        return

    # ── Standard recommendation flow ─────────
    if recommend_clicked:
        if not movie_title:
            st.warning("Pick a movie above first, or hit Surprise me.")
            return

        with st.spinner("Matching taste profiles…"):
            recs = _recommend(movie_title)

        if genre_filter and not recs.empty:
            mask = recs["genres"].apply(
                lambda g: any(genre in g.split("|") for genre in genre_filter)
            )
            filtered = recs[mask]
            if not filtered.empty:
                recs = filtered

        if recs.empty:
            st.error("No matches found for this title. Try another movie.")
            return

        display = re.sub(r"\s*\(\d{4}\)\s*$", "", movie_title).strip()
        section_title(f"Because you liked <em>{display}</em>", count=len(recs))
        method = "genre similarity + community ratings (KNN)" if (use_hybrid and knn_model is not None) else "genre similarity"
        st.caption(f"{len(recs)} titles ranked by {method}")
        render_cards_grid(recs, cards_per_row=5)
    else:
        render_trending_strip(trending_df, movies_df)
        render_mood_explorer(clustered_df, cluster_names)
        st.markdown(
            """
            <div style="
              text-align:center;
              padding: 32px 24px 48px;
              color: var(--cream-muted);
              font-family: 'Inter', sans-serif;
              font-size: 1rem;
              letter-spacing: 0.01em;
            ">
              Or search for a film above and click <strong style="color:var(--cream-dim)">Get recommendations</strong>
              &mdash; or try <strong style="color:var(--cream-dim)">Surprise me</strong> for a curated pick.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Footer ────────────────────────────────
    st.markdown(
        """
        <div class="mfy-footer">
          <div class="mfy-footer-logo">Movie<em>fy</em></div>
          <div class="mfy-footer-sub">Content-based filtering &middot; KNN collaborative filtering &middot; KMeans clustering &middot; OMDb API</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
