from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pickle
import json
import collections
import __main__
from main import CFRecommender

__main__.CFRecommender = CFRecommender

app       = FastAPI()
templates = Jinja2Templates(directory="public")


POPULAR_MIN_COUNT = 20

ITEM_FILE   = "dataset/ml-100k (1)/ml-100k/u.item"
POSTER_FILE = "posters.json"
MODEL_FILE  = "recommender_models.pkl"

with open(POSTER_FILE) as f:
    posters = {int(k): v for k, v in json.load(f).items()}

movies = {}
with open(ITEM_FILE, encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        movies[int(parts[0])] = parts[1]

with open(MODEL_FILE, "rb") as f:
    _saved     = pickle.load(f)
    user_model = _saved["user_model"]
    item_model = _saved["item_model"]


def get_popular_items(n: int = 10):
    Y          = user_model.Y_data
    item_stats = collections.defaultdict(list)
    for _, item_0based, rating in Y:
        item_stats[int(item_0based)].append(float(rating))

    popular = [
        (item_0based, np.mean(ratings))
        for item_0based, ratings in item_stats.items()
        if len(ratings) >= POPULAR_MIN_COUNT
    ]
    popular.sort(key=lambda x: x[1], reverse=True)

    return [
        {
            "title":  movies.get(item_0based + 1, f"Movie {item_0based + 1}"),
            "score":  round(score, 2),
            "poster": posters.get(item_0based + 1),
        }
        for item_0based, score in popular[:n]
    ]


def get_user_history(user_id: int, n: int = 10):
    if not (0 <= user_id < user_model.n_users):
        return []

    Y   = user_model.Y_data
    ids = np.where(Y[:, 0] == user_id)[0]

    history = []
    for idx in ids[:n]:
        item_0based = int(Y[idx, 1])
        rating      = float(Y[idx, 2])
        movie_id    = item_0based + 1
        history.append({
            "title":  movies.get(movie_id, f"Movie {movie_id}"),
            "rating": rating,
            "poster": posters.get(movie_id),
        })

    history.sort(key=lambda x: x["rating"], reverse=True)
    return history



def get_user_recommendations(user_id: int, n: int = 10):
    """
    - User không tồn tại hoặc chưa có rating → global popular.
    - Có rating nhưng không tìm được neighbor (do min_common) → global popular.
    - Còn lại → User-based CF.
    """
    if not (0 <= user_id < user_model.n_users):
        return get_popular_items(n), True

    items_u, _ = user_model.get_sparse_row(user_id)
    if len(items_u) == 0:
        return get_popular_items(n), True

    recs = user_model.recommend(user_id, n_rec=n)

    if not recs or np.std([s for _, s in recs]) < 0.01:
        return get_popular_items(n), True

    result = []
    for item_0based, score in recs:
        movie_id      = item_0based + 1
        display_score = float(np.clip(score, 1.0, 5.0))
        result.append({
            "title":  movies.get(movie_id, f"Movie {movie_id}"),
            "score":  round(display_score, 2),
            "poster": posters.get(movie_id),
        })
    return result, False


def get_item_recommendations(user_id: int, n: int = 10):
    if not (0 <= user_id < item_model.n_users):
        return get_popular_items(n), True

    ids = np.where(item_model.Y_data[:, 0] == user_id)[0]
    if len(ids) == 0:
        return get_popular_items(n), True

    rated_set        = set(item_model.Y_data[ids, 1].astype(int))
    candidate_scores = collections.defaultdict(float)

    for item_0based in list(rated_set)[:10]:
        if item_0based >= item_model.S.shape[0]:
            continue
        sim_row     = item_model.S[item_0based]
        top_indices = np.argsort(sim_row)[::-1]

        count = 0
        for idx in top_indices:
            if int(idx) in rated_set:
                continue
            if sim_row[idx] <= 0:
                break
            candidate_scores[int(idx)] += float(sim_row[idx])
            count += 1
            if count >= 30:
                break

    if not candidate_scores:
        return get_popular_items(n), True

    sorted_candidates = sorted(
        candidate_scores.items(), key=lambda x: x[1], reverse=True
    )

    result = []
    for item_0based, score in sorted_candidates[:n]:
        movie_id = item_0based + 1
        result.append({
            "title":  movies.get(movie_id, f"Movie {movie_id}"),
            "score":  round(score, 4),
            "poster": posters.get(movie_id),
        })
    return result, False


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/recommend", response_class=HTMLResponse)
async def recommend(
    request: Request,
    user_id: int = 0,
    method:  str = "both",
    n:       int = 10,
):
    user_recs, user_cold = (
        get_user_recommendations(user_id, n)
        if method in ["user", "both"]
        else ([], False)
    )
    item_recs, item_cold = (
        get_item_recommendations(user_id, n)
        if method in ["item", "both"]
        else ([], False)
    )

    history = get_user_history(user_id, n)

    return templates.TemplateResponse(request, "index.html", {
        "user_id":   user_id,
        "method":    method,
        "user_recs": user_recs,
        "item_recs": item_recs,
        "history":   history,
        "n":         n,
        "user_cold": user_cold,
        "item_cold": item_cold,
    })