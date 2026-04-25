import pickle
import numpy as np
from recommendation import UserBasedCF

# ================= LOAD MODEL =================
def load_model(path="usercf_model.pkl"):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model


model = load_model("user_cf.pkl")

print("✅ Model loaded successfully")
print("Users:", model.n_users)
print("Items:", model.n_items)


# ================= TEST RECOMMEND =================
def recommend_user(user_id, top_n=10):
    print(f"\n🎬 Top {top_n} recommendations for User {user_id}:\n")

    recs = model.recommend(user_id, n_rec=top_n)

    if len(recs) == 0:
        print("No recommendations found!")
        return

    for idx, (item_id, score) in enumerate(recs):
        print(f"{idx+1}. Item {item_id} | Score: {score:.4f}")


# ================= INTERACTIVE MODE =================
while True:
    try:
        user_id = int(input("\nEnter user_id (-1 to exit): "))

        if user_id == -1:
            break

        if user_id >= model.n_users:
            print("❌ Invalid user_id")
            continue

        recommend_user(user_id)

    except Exception as e:
        print("Error:", e)