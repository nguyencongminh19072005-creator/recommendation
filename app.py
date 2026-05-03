from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import pickle
import json
import __main__
from recommendation import UserBasedCF
from itembased import FastItemCF_Scratch

# Hack để fix lỗi pickle không tìm thấy class khi load (do model được train trên Colab/Jupyter/Local file chính)
__main__.UserBasedCF = UserBasedCF
__main__.FastItemCF_Scratch = FastItemCF_Scratch

app = FastAPI()
templates = Jinja2Templates(directory="public")

# ── Load posters ──────────────────────────────────────────
with open("posters.json") as f:
    posters = json.load(f)
    posters = {int(k): v for k, v in posters.items()}

# ── Load movie names ──────────────────────────────────────
movies = {}
with open("dataset/ml-100k (1)/ml-100k/u.item", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        movies[int(parts[0])] = parts[1]

# ── Load models ───────────────────────────────────────────
with open("user_based_cf.pkl", "rb") as f:
    user_model = pickle.load(f)

with open("fast_item_cf.pkl", "rb") as f:
    item_model = pickle.load(f)

# ── Helpers ───────────────────────────────────────────────
def get_popular_items(n: int = 10):
    """Fallback cold start: top phim được rate nhiều và điểm cao"""
    Y = user_model.Y_data
    item_stats = {}
    for u, i, r in Y:
        item_id = int(i)
        if item_id not in item_stats:
            item_stats[item_id] = []
        item_stats[item_id].append(r)

    popular = []
    for item_id, ratings in item_stats.items():
        if len(ratings) >= 20:
            avg = np.mean(ratings)
            popular.append((item_id, avg))

    popular.sort(key=lambda x: x[1], reverse=True)

    result = []
    for item_id_0based, score in popular[:n]:
        movie_id = item_id_0based + 1
        title  = movies.get(movie_id, f"Movie {movie_id}")
        poster = posters.get(movie_id)
        result.append({"title": title, "score": round(float(score), 2), "poster": poster})
    return result


def get_user_recommendations(user_id: int, n: int = 10):
    # 🔥 FIX 1: user ngoài range
    if user_id >= user_model.n_users or user_id < 0:
        return get_popular_items(n), True

    items_u, _ = user_model.get_sparse_row(user_id)

    # 🔥 FIX 2: user không có rating
    if len(items_u) == 0:
        return get_popular_items(n), True

    recs = user_model.recommend(user_id, n_rec=n)
    result = []
    for item_id_0based, score in recs:
        movie_id = item_id_0based + 1
        title  = movies.get(movie_id, f"Movie {movie_id}")
        poster = posters.get(movie_id)
        
        # Clip điểm số hiển thị trên Web nằm trong khoảng 1.0 đến 5.0
        display_score = min(max(float(score), 1.0), 5.0) if score is not None else None
        
        result.append({
            "title": title,
            "score": round(display_score, 2) if display_score else None,
            "poster": poster
        })
    return result, False


def get_item_recommendations(user_id: int, n: int = 10):
    # 🔥 FIX 1: user ngoài range (0-based)
    if user_id >= item_model.n_users or user_id < 0:
        return get_popular_items(n), True

    # 🔥 FIX 2: kiểm tra user có rating nào chưa
    ids = np.where(item_model.Y_data[:, 0] == user_id)[0]
    if len(ids) == 0:
        return get_popular_items(n), True

    recs = item_model.recommend(user_id, top_k=n, exclude_rated=True)

    if not recs:
        return get_popular_items(n), True

    result = []
    for item_id_0based in recs:
        movie_id = int(item_id_0based) + 1
        title  = movies.get(movie_id, f"Movie {movie_id}")
        poster = posters.get(movie_id)
        result.append({
            "title": title,
            "score": None,
            "poster": poster
        })

    return result, False


def get_user_history(user_id: int, n: int = 10):
    # 🔥 FIX: user ngoài range
    if user_id >= user_model.n_users or user_id < 0:
        return []

    Y = user_model.Y_data
    ids = np.where(Y[:, 0] == user_id)[0]

    history = []
    for idx in ids[:n]:
        item_id_0based = int(Y[idx, 1])
        rating = float(Y[idx, 2])
        movie_id = item_id_0based + 1
        title  = movies.get(movie_id, f"Movie {movie_id}")
        poster = posters.get(movie_id)
        history.append({
            "title": title,
            "rating": rating,
            "poster": poster
        })

    history.sort(key=lambda x: x["rating"], reverse=True)
    return history


# ── Routes ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "index.html")

@app.get("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, user_id: int = 0, method: str = "both", n: int = 10):
    user_recs, user_cold = get_user_recommendations(user_id, n) if method in ["user", "both"] else ([], False)
    item_recs, item_cold = get_item_recommendations(user_id, n) if method in ["item", "both"] else ([], False)
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
