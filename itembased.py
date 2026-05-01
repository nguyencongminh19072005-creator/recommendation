import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from google.colab import drive


class FastItemCF:
    def __init__(self, Y_data, n_users, n_items, k=20, shrink=20, min_common=15, pop_weight=0.0):
        self.Y_data = Y_data
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.shrink = shrink
        self.min_common = min_common
        self.pop_weight = pop_weight

    def fit(self):
        self.normalize()
        self.similarity()

    def normalize(self):
        # Sort theo (item, user) → CSR by item nhất quán
        sort_idx    = np.lexsort((self.Y_data[:, 0], self.Y_data[:, 1]))
        self.Y_data = self.Y_data[sort_idx]

        self.mu = np.zeros(self.n_users)
        users   = self.Y_data[:, 0].astype(int)
        for u in range(self.n_users):
            ids = np.where(users == u)[0]
            self.mu[u] = np.mean(self.Y_data[ids, 2]) if len(ids) > 0 else 0

        rows    = self.Y_data[:, 1].astype(int)   # item là row
        cols    = self.Y_data[:, 0].astype(int)   # user là col
        ratings = self.Y_data[:, 2]
        centered = ratings - self.mu[cols]

        # Build CSR theo item
        self.indptr   = np.zeros(self.n_items + 1, dtype=int)
        for r in rows:
            self.indptr[r + 1] += 1
        self.indptr = np.cumsum(self.indptr)

        self.indices  = np.zeros(len(cols), dtype=int)
        self.csr_data = np.zeros(len(centered))

        tracker = self.indptr[:-1].copy()
        for idx in range(len(centered)):
            r   = rows[idx]
            pos = tracker[r]
            self.indices[pos]  = cols[idx]
            self.csr_data[pos] = centered[idx]
            tracker[r] += 1

        # item popularity (giữ nguyên)
        item_counts        = np.bincount(rows, minlength=self.n_items)
        self.item_pop      = np.log1p(item_counts)
        self.item_pop_norm = self.item_pop / (np.max(self.item_pop) + 1e-8)

    def get_sparse_row(self, i):
        """Trả về (user_ids, ratings_bar) của item i."""
        start = self.indptr[i]
        end   = self.indptr[i + 1]
        return self.indices[start:end], self.csr_data[start:end]

    def get_user_items(self, u):
        """Trả về (item_ids, ratings_bar) mà user u đã rate."""
        mask  = self.Y_data[:, 0].astype(int) == u
        items = self.Y_data[mask, 1].astype(int)
        rbar  = self.Y_data[mask, 2] - self.mu[u]
        return items, rbar

    def similarity(self):
        self.S = np.zeros((self.n_items, self.n_items))

        for i in range(self.n_items):
            users_i, ratings_i = self.get_sparse_row(i)
            if len(users_i) == 0:
                continue

            for j in range(i, self.n_items):
                if i == j:
                    self.S[i, j] = 1.0
                    continue

                users_j, ratings_j = self.get_sparse_row(j)
                if len(users_j) == 0:
                    continue

                common, idx_i, idx_j = np.intersect1d(
                    users_i, users_j, return_indices=True
                )
                if len(common) < self.min_common:
                    continue

                r_i   = ratings_i[idx_i]
                r_j   = ratings_j[idx_j]
                denom = np.sqrt(np.sum(r_i**2)) * np.sqrt(np.sum(r_j**2))
                if denom == 0:
                    continue

                sim  = np.sum(r_i * r_j) / denom
                sim *= len(common) / (len(common) + self.shrink)
                sim  = max(sim, 0)

                self.S[i, j] = self.S[j, i] = sim

    def predict(self, u, i):
        if u >= self.n_users or i >= self.n_items:
            return None

        items_u, rbar_u = self.get_user_items(u)
        if len(items_u) == 0:
            return None

        sims     = self.S[i, items_u]
        top_idx  = np.argsort(sims)[::-1][:self.k]
        top_sims = np.maximum(sims[top_idx], 0)
        top_rbar = rbar_u[top_idx]

        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return None

        pred = self.mu[u] + np.sum(top_sims * top_rbar) / denom
        return float(np.clip(pred, 1, 5))

    def predict_score_for_ranking(self, u, i):
        items_u, rbar_u = self.get_user_items(u)
        if len(items_u) == 0:
            return None

        sims     = self.S[i, items_u]
        top_idx  = np.argsort(sims)[::-1][:self.k]
        top_sims = np.maximum(sims[top_idx], 0)
        top_rbar = rbar_u[top_idx]

        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return None

        pop_bias = self.pop_weight * self.item_pop_norm[i]
        return float(self.mu[u] + np.sum(top_sims * top_rbar) / denom + pop_bias)

    def recommend(self, u, n_rec=5):
        if u >= self.n_users:
            top_idx = np.argsort(self.item_pop_norm)[::-1][:n_rec]
            return [(int(i), float(self.item_pop_norm[i])) for i in top_idx]

        items_u, _ = self.get_user_items(u)

        if len(items_u) == 0:
            top_idx = np.argsort(self.item_pop_norm)[::-1][:n_rec]
            return [(int(i), float(self.item_pop_norm[i])) for i in top_idx]

        unrated = np.setdiff1d(np.arange(self.n_items), items_u)
        preds   = [(int(i), p) for i in unrated
                   if (p := self.predict_score_for_ranking(u, i)) is not None]

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_rec]

# ======================================================================
# Split data
# ======================================================================
def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
    """Chia dữ liệu theo per-user + timestamp."""
    user_items = {}
    for u, i, r, t in Y:
        user_items.setdefault(int(u), []).append((i, r, t))

    train, valid, test = [], [], []
    for u in user_items:
        items = sorted(user_items[u], key=lambda x: x[2])  # sort theo timestamp
        n = len(items)
        n_train = int(train_ratio * n)
        n_valid = int(valid_ratio * n)
        for i, r, _ in items[:n_train]:
            train.append([u, i, r])
        for i, r, _ in items[n_train:n_train + n_valid]:
            valid.append([u, i, r])
        for i, r, _ in items[n_train + n_valid:]:
            test.append([u, i, r])

    return np.array(train), np.array(valid), np.array(test)


# ======================================================================
# RMSE
# ======================================================================
def rmse(model, data):
    se, cnt = 0, 0
    for u, i, r in data:
        pred = model.predict(int(u), int(i))
        if pred is None:
            continue
        se  += (r - pred) ** 2
        cnt += 1
    return np.sqrt(se / cnt) if cnt > 0 else float('nan')


# ======================================================================
# Evaluate Precision & Recall
# ======================================================================
def evaluate_top_k(model, data, n_items, K=10, threshold=4, n_neg=300):
    np.random.seed(42)
    user_liked = {}
    for u, i, r in data:
        if r >= threshold:
            user_liked.setdefault(int(u), set()).add(int(i))

    precisions, recalls = [], []

    for u in user_liked:
        liked       = user_liked[u]
        items_u, _  = model.get_sparse_row(u)
        items_u     = set(items_u.tolist())

        valid_liked = liked - items_u
        if not valid_liked:
            continue

        all_items = set(range(n_items))
        negatives = list(all_items - items_u - valid_liked)
        if len(negatives) < n_neg:
            continue

        negatives  = np.random.choice(negatives, n_neg, replace=False)
        candidates = list(valid_liked) + list(negatives)
        np.random.shuffle(candidates)  # Fix tie-breaking bug

        preds = []
        for i in candidates:
            p = model.predict_score_for_ranking(u, i)
            if p is not None:
                preds.append((i, p))

        if not preds:
            continue

        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = set([i for i, _ in preds[:K]])
        hits  = len(top_k & valid_liked)

        precisions.append(hits / K)
        recalls.append(hits / len(valid_liked))

    return np.mean(precisions), np.mean(recalls)


# ======================================================================
# MAIN
# ======================================================================
if __name__ == "__main__":
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/movielen/'

    df = pd.read_csv(BASE_PATH + 'ml-100k (1)/ml-100k/u.data',
                     sep='\t', names=['u', 'i', 'r', 't'])
    df['u'] -= 1
    df['i'] -= 1
    Y = df.values

    n_users = int(np.max(Y[:, 0])) + 1
    n_items = int(np.max(Y[:, 1])) + 1

    Y_train, Y_valid, Y_test = split_data(Y)

    # ===== TUNING TRÊN VALID =====
    best_val_p  = 0
    best_params = {}
    best_model  = None

    for k in [20, 50, 100]:
        for shrink in [10, 25, 50]:
            for min_common in [2, 5, 10]:
                model = FastItemCF(Y_train, n_users, n_items,
                                   k=k, shrink=shrink, min_common=min_common)
                model.fit()
                p, r = evaluate_top_k(model, Y_valid, n_items, K=10)
                print(f"k={k}, shrink={shrink}, min_common={min_common} "
                      f"-> P@10={p:.4f}, R@10={r:.4f}")

                if p > best_val_p:
                    best_val_p  = p
                    best_params = {'k': k, 'shrink': shrink, 'min_common': min_common}
                    best_model  = model

    print(f"\nBest params: {best_params}")

    # ===== ĐÁNH GIÁ TRÊN TEST (1 lần duy nhất) =====
    p_test, r_test = evaluate_top_k(best_model, Y_test, n_items, K=10)
    print(f"Test P@10 : {p_test:.4f}")
    print(f"Test R@10 : {r_test:.4f}")

    # ===== LEARNING CURVE: Train RMSE vs Valid RMSE =====
    print("\nĐang vẽ learning curve (Train RMSE vs Valid RMSE)...")

    train_sizes  = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    train_rmses  = []
    valid_rmses  = []

    np.random.seed(42)
    for frac in train_sizes:
        n = int(len(Y_train) * frac)

        # Random sample — không cắt theo thứ tự
        idx   = np.random.choice(len(Y_train), n, replace=False)
        Y_sub = Y_train[idx]

        lc_model = FastItemCF(Y_sub, n_users, n_items, **best_params)
        lc_model.fit()

        tr = rmse(lc_model, Y_sub)    # Train RMSE
        va = rmse(lc_model, Y_valid)  # Valid RMSE ✅ (không dùng test)

        train_rmses.append(tr)
        valid_rmses.append(va)
        print(f"  Size {frac*100:.0f}%: Train RMSE={tr:.4f}, Valid RMSE={va:.4f}")

    # ===== VẼ BIỂU ĐỒ =====
    size_labels = [f"{int(s*100)}%" for s in train_sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(size_labels, train_rmses, 'o-',  label='Train RMSE')
    plt.plot(size_labels, valid_rmses, 'o--', label='Valid RMSE')
    plt.xlabel('Training data size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve - Item-Based CF')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'learning_curve_item.png', dpi=150)
    plt.show()

    # ===== FINAL MODEL =====
    final_model = FastItemCF(Y[:, :3], n_users, n_items, **best_params)
    final_model.fit()

    with open(BASE_PATH + "fast_item_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print("Saved fast_item_cf.pkl to Drive")