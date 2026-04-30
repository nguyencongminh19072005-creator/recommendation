import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from google.colab import drive

class UserBasedCF:
    def __init__(self, Y_data, n_users, n_items, k=20, shrink=20, min_common=15):
        self.Y_data = Y_data
        self.k = k
        self.n_users = n_users
        self.n_items = n_items
        self.shrink = shrink
        self.min_common = min_common

    def normalize(self):
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros(self.n_users)
        users = self.Y_data[:, 0]

        for u in range(self.n_users):
            ids = np.where(users == u)[0]
            ratings = self.Y_data[ids, 2]
            mean = np.mean(ratings) if len(ratings) > 0 else 0
            self.mu[u] = mean
            self.Ybar_data[ids, 2] = ratings - mean

        rows = self.Ybar_data[:, 0].astype(int)
        cols = self.Ybar_data[:, 1].astype(int)
        data = self.Ybar_data[:, 2]

        self.indptr = np.zeros(self.n_users + 1, dtype=int)
        for r in rows:
            self.indptr[r + 1] += 1
        self.indptr = np.cumsum(self.indptr)

        self.indices  = np.zeros(len(cols), dtype=int)
        self.csr_data = np.zeros(len(data))

        tracker = self.indptr[:-1].copy()
        for i in range(len(data)):
            r   = rows[i]
            pos = tracker[r]
            self.indices[pos]  = cols[i]
            self.csr_data[pos] = data[i]
            tracker[r] += 1

    def get_sparse_row(self, u):
        start = self.indptr[u]
        end   = self.indptr[u + 1]
        return self.indices[start:end], self.csr_data[start:end]

    def similarity(self):
        self.S = np.zeros((self.n_users, self.n_users))

        for i in range(self.n_users):
            items_i, ratings_i = self.get_sparse_row(i)

            for j in range(i, self.n_users):
                if i == j:
                    self.S[i, j] = 1.0
                    continue

                items_j, ratings_j = self.get_sparse_row(j)
                common, idx_i, idx_j = np.intersect1d(
                    items_i, items_j, return_indices=True
                )

                if len(common) < self.min_common:
                    self.S[i, j] = self.S[j, i] = 0
                    continue

                r_i = ratings_i[idx_i]
                r_j = ratings_j[idx_j]

                denom = np.sqrt(np.sum(r_i**2)) * np.sqrt(np.sum(r_j**2))
                sim   = np.sum(r_i * r_j) / denom if denom != 0 else 0

                shrink  = len(common) / (len(common) + self.shrink)
                sim    *= shrink

                self.S[i, j] = self.S[j, i] = sim

    def fit(self):
        self.normalize()
        self.similarity()

    def predict(self, u, i):
        if u >= self.n_users:
            return None

        item_indices = np.where(self.Y_data[:, 1] == i)[0]
        if len(item_indices) == 0:
            return None

        users_rated_i = self.Y_data[item_indices, 0].astype(int)
        
        # 🟢 FIX LỖI LEAKAGE DỮ LIỆU Ở ĐÂY:
        # Loại bỏ chính user `u` ra khỏi danh sách những người đã đánh giá phim `i`
        users_rated_i = users_rated_i[users_rated_i != u]
        
        if len(users_rated_i) == 0:
            return self.mu[u] # Nếu không có ai khác đánh giá, trả về mức trung bình của user

        sims = self.S[u, users_rated_i]

        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_users = users_rated_i[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        ratings = []
        for user in top_users:
            items_u, ratings_u = self.get_sparse_row(user)
            idx = np.where(items_u == i)[0]
            ratings.append(ratings_u[idx[0]] if len(idx) > 0 else 0)

        ratings = np.array(ratings)
        denom   = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.mu[u] # Fallback về trung bình nếu tổng độ tương đồng = 0

        pred = self.mu[u] + np.sum(top_sims * ratings) / denom
        return np.clip(pred, 1, 5)

    def predict_score_for_ranking(self, u, i):
        item_indices = np.where(self.Y_data[:, 1] == i)[0]
        if len(item_indices) == 0:
            return None

        users_rated_i = self.Y_data[item_indices, 0].astype(int)
        
        # 🟢 FIX LỖI LEAKAGE DỮ LIỆU Ở ĐÂY:
        users_rated_i = users_rated_i[users_rated_i != u]
        
        if len(users_rated_i) == 0:
            return self.mu[u]

        sims = self.S[u, users_rated_i]

        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_users = users_rated_i[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        ratings = []
        for user in top_users:
            items_u, ratings_u = self.get_sparse_row(user)
            idx = np.where(items_u == i)[0]
            ratings.append(ratings_u[idx[0]] if len(idx) > 0 else 0)

        ratings = np.array(ratings)
        denom   = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.mu[u]

        return self.mu[u] + np.sum(top_sims * ratings) / denom

    def recommend(self, u, n_rec=5):
        items_u, _ = self.get_sparse_row(u)
        all_items  = np.arange(self.n_items)
        unrated    = np.setdiff1d(all_items, items_u)

        preds = []
        for i in unrated:
            p = self.predict_score_for_ranking(u, i)
            if p is not None:
                preds.append((i, p))

        preds.sort(key=lambda x: x[1], reverse=True)
        return preds[:n_rec]


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


def rmse(model, data):
    se, cnt = 0, 0
    for u, i, r in data:
        pred = model.predict(int(u), int(i))
        if pred is None:
            continue
        se  += (r - pred) ** 2
        cnt += 1
    return np.sqrt(se / cnt) if cnt > 0 else float('nan')


def evaluate_top_k(model, data, n_items, K=10, threshold=4, n_neg=300):
    np.random.seed(42)
    user_liked = {}
    for u, i, r in data:
        if r >= threshold:
            user_liked.setdefault(int(u), set()).add(int(i))

    precisions, recalls = [], []

    for u in user_liked:
        liked      = user_liked[u]
        items_u, _ = model.get_sparse_row(u)
        items_u    = set(items_u)

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

    # ===== TUNING — chọn best params theo VALID RMSE thấp nhất =====
    best_precision = -1
    best_params = {}
    best_model = None

    # 🟢 ĐIỀU CHỈNH: Tăng giới hạn K láng giềng lên để phù hợp với 100k data
    for k in [40, 60, 80]:
        for shrink in [10, 20, 30]:
            for min_common in [5, 10]:
                model = UserBasedCF(Y_train, n_users, n_items,
                                    k=k, shrink=shrink, min_common=min_common)
                model.fit()

                # Dùng Precision thay vì RMSE
                p_val, r_val = evaluate_top_k(model, Y_valid, n_items, K=10)

                if not np.isnan(p_val) and p_val > best_precision:
                    best_precision = p_val
                    best_params = {
                        'k': k,
                        'shrink': shrink,
                        'min_common': min_common
                    }
                    best_model = model

                print(f"k={k}, shrink={shrink}, min_common={min_common} "
                      f"-> P@10={p_val:.4f}, R@10={r_val:.4f}")
    
    print(f"\nBest params: {best_params}")
    print(f"Best Precision@10: {best_precision:.4f}")

    # ===== ĐÁNH GIÁ TRÊN TEST (1 lần duy nhất) =====
    test_rmse = rmse(best_model, Y_test)
    p_test, r_test = evaluate_top_k(best_model, Y_test, n_items, K=10)
    print(f"Test RMSE : {test_rmse:.4f}")
    print(f"Test P@10 : {p_test:.4f}")
    print(f"Test R@10 : {r_test:.4f}")

    # ===== LEARNING CURVE: Train RMSE vs Valid RMSE =====
    print("\nĐang vẽ learning curve...")
    train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
    train_rmses = []
    valid_rmses = []
    valid_precisions = []
    np.random.seed(42)
    
    for frac in train_sizes:
        n     = int(len(Y_train) * frac)
        idx   = np.random.choice(len(Y_train), n, replace=False)
        Y_sub = Y_train[idx]

        lc_model = UserBasedCF(Y_sub, n_users, n_items, **best_params)
        lc_model.fit()

        tr = rmse(lc_model, Y_sub)
        va = rmse(lc_model, Y_valid)
        train_rmses.append(tr)
        valid_rmses.append(va)
        p_val, _ = evaluate_top_k(lc_model, Y_valid, n_items, K=10)
        p_val = 0 if np.isnan(p_val) else p_val
        valid_precisions.append(p_val)
        print(f"Size {frac*100:.0f}%: "
        f"Train={tr:.4f}, Valid={va:.4f}, "
        f"P@10={p_val:.4f}, Gap={va-tr:.4f}")
        
    # ===== VẼ BIỂU ĐỒ =====
    size_labels = [f"{int(s*100)}%" for s in train_sizes]

    plt.figure(figsize=(8, 5))
    plt.plot(size_labels, train_rmses, 'o-',  label='Train RMSE')
    plt.plot(size_labels, valid_rmses, 'o--', label='Valid RMSE')
    plt.xlabel('Training data size')
    plt.ylabel('RMSE')
    plt.title('Learning Curve - User-Based CF')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'learning_curve_user.png', dpi=150)
    plt.show()
    
    plt.figure(figsize=(8, 5))
    plt.plot(size_labels, valid_precisions, 'o-', label='Valid Precision@10')
    plt.xlabel('Training data size')
    plt.ylabel('Precision@10')
    plt.title('Learning Curve - Recommendation Quality (Precision@10)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(BASE_PATH + 'learning_curve_precision.png', dpi=150)
    plt.show()
    
    # ===== FINAL MODEL =====
    final_model = UserBasedCF(Y, n_users, n_items, **best_params)
    final_model.fit()

    with open(BASE_PATH + "user_based_cf.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"Saved model. Best params: {best_params}")