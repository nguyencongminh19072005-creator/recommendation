import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from google.colab import drive


class FastItemCF:
    def __init__(self, Y_data, n_users, n_items, k=50, shrink=10, min_common=2, pop_weight=0.1):
        self.Y_data = Y_data.astype(np.float64)
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.shrink = shrink
        self.min_common = min_common
        self.pop_weight = pop_weight

    def normalize(self):
        self.mu = np.zeros(self.n_users)
        for u in range(self.n_users):
            idx = self.Y_data[:, 0] == u
            if np.any(idx):
                self.mu[u] = np.mean(self.Y_data[idx, 2])

        users   = self.Y_data[:, 0].astype(np.int32)
        items   = self.Y_data[:, 1].astype(np.int32)
        ratings = self.Y_data[:, 2].astype(np.float64)

        item_counts = np.bincount(items, minlength=self.n_items)
        self.item_pop      = np.log1p(item_counts)
        self.item_pop_norm = self.item_pop / (np.max(self.item_pop) + 1e-8)

        centered = ratings - self.mu[users]
        self.Ybar = sparse.coo_matrix(
            (centered, (users, items)),
            shape=(self.n_users, self.n_items)
        ).tocsr()

    def similarity(self):
        S = cosine_similarity(self.Ybar.T, dense_output=False).toarray()

        binary      = self.Ybar.copy()
        binary.data = np.ones_like(binary.data)
        co_count    = (binary.T @ binary).toarray()

        S = S * (co_count / (co_count + self.shrink))
        S[co_count < self.min_common] = 0
        S[S < 0] = 0
        np.fill_diagonal(S, 0)
        self.S = S

    def fit(self):
        self.normalize()
        self.similarity()

    def get_sparse_row(self, u):
        row = self.Ybar[u]
        return row.indices, row.data

    def predict(self, u, i):
        if u >= self.n_users or i >= self.n_items:
            return None
        item_indices = np.where(self.Y_data[:, 1] == i)[0]
        if len(item_indices) == 0:
            return None
        items_u, _ = self.get_sparse_row(u)
        if len(items_u) == 0:
            return None

        sims      = self.S[i, items_u]
        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_items = items_u[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        rated_vals = self.Ybar[u, top_items].toarray().flatten()
        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return None

        pred = self.mu[u] + np.sum(top_sims * rated_vals) / denom
        return float(np.clip(pred, 1, 5))

    def predict_score_for_ranking(self, u, i):
        if u >= self.n_users or i >= self.n_items:
            return None
        items_u, _ = self.get_sparse_row(u)
        if len(items_u) == 0:
            return None

        sims      = self.S[i, items_u]
        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_items = items_u[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        rated_vals = self.Ybar[u, top_items].toarray().flatten()
        denom = np.sum(np.abs(top_sims))
        if denom == 0:
            return None

        pop_bias = self.pop_weight * self.item_pop_norm[i]
        return float(self.mu[u] + np.sum(top_sims * rated_vals) / denom + pop_bias)

    def recommend(self, u, n_rec=5):
        items_u, _ = self.get_sparse_row(u)
        all_items  = np.arange(self.n_items)
        unrated    = np.setdiff1d(all_items, items_u)

        if len(unrated) == 0:
            return []

        if len(items_u) == 0:
            pop_scores = self.item_pop_norm[unrated]
            top_idx = np.argsort(pop_scores)[::-1][:n_rec]
            return [(int(unrated[i]), float(pop_scores[i])) for i in top_idx]

        preds = []
        for i in unrated:
            p = self.predict_score_for_ranking(u, i)
            if p is not None:
                preds.append((int(i), p))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_rec]


# ======================================================================
# Split data
# ======================================================================
def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
    np.random.seed(42)
    user_items = {}
    for u, i, r in Y:
        user_items.setdefault(int(u), []).append((i, r))

    train, valid, test = [], [], []
    for u in user_items:
        items = user_items[u]
        np.random.shuffle(items)
        n = len(items)
        n_train = int(train_ratio * n)
        n_valid = int(valid_ratio * n)
        for i, r in items[:n_train]:
            train.append([u, i, r])
        for i, r in items[n_train:n_train + n_valid]:
            valid.append([u, i, r])
        for i, r in items[n_train + n_valid:]:
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
    df = df.drop('t', axis=1)
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
    final_model = FastItemCF(Y, n_users, n_items, **best_params)
    final_model.fit()

    with open(BASE_PATH + "fast_item_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print("Saved fast_item_cf.pkl to Drive")