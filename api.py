import requests
import pandas as pd
import json
import time

API_KEY = "c7ad753849b35c83be5394a646f70425"  
BASE_URL = "https://api.themoviedb.org/3"
IMG_BASE = "https://image.tmdb.org/t/p/w300"

# Đọc tên phim từ u.item
movies = {}
with open("dataset/ml-100k (1)/ml-100k/u.item", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("|")
        movie_id = int(parts[0])
        title_year = parts[1]  # vd: "Toy Story (1995)"
        # Tách tên và năm
        if "(" in title_year:
            title = title_year[:title_year.rfind("(")].strip()
            year  = title_year[title_year.rfind("(")+1:title_year.rfind(")")]
        else:
            title = title_year.strip()
            year  = ""
        movies[movie_id] = {"title": title_year, "search": title, "year": year}

# Fetch poster từ TMDB
posters = {}
for movie_id, info in movies.items():
    try:
        params = {
            "api_key": API_KEY,
            "query": info["search"],
            "year": info["year"],
            "language": "en-US"
        }
        res = requests.get(f"{BASE_URL}/search/movie", params=params)
        data = res.json()

        if data["results"]:
            poster_path = data["results"][0].get("poster_path")
            posters[movie_id] = IMG_BASE + poster_path if poster_path else None
        else:
            posters[movie_id] = None

        time.sleep(0.05)  # tránh rate limit

    except Exception as e:
        posters[movie_id] = None
        print(f"Error movie {movie_id}: {e}")

    if movie_id % 100 == 0:
        print(f"Done {movie_id}/1682")

# Lưu ra file json
with open("posters.json", "w") as f:
    json.dump(posters, f)

print(f"Done! Saved {sum(1 for v in posters.values() if v)} / {len(posters)} posters")