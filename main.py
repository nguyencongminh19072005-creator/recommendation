import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ==========================================
# 1. UNIFIED KNN RECOMMENDER CLASS
# ==========================================
class CFRecommender:
    def __init__(self, Y_data, n_users, n_items, k=20, shrink=20, min_common=15, mode='user'):
        self.Y_data = Y_data
        self.n_users = n_users
        self.n_items = n_items
        self.k = k
        self.shrink = shrink
        self.min_common = min_common
        self.mode = mode.lower()

    def normalize(self):
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros(self.n_users)
        users = self.Y_data[:, 0].astype(int)

        for u in range(self.n_users):
            ids = np.where(users == u)[0]
            if len(ids) > 0:
                mean = np.mean(self.Y_data[ids, 2])
                self.mu[u] = mean
                self.Ybar_data[ids, 2] = self.Y_data[ids, 2] - mean

        if self.mode == 'user':
            rows = self.Ybar_data[:, 0].astype(int)
            cols = self.Ybar_data[:, 1].astype(int)
            self.n_elements = self.n_users
        else:
            rows = self.Ybar_data[:, 1].astype(int)
            cols = self.Ybar_data[:, 0].astype(int)
            self.n_elements = self.n_items

        data = self.Ybar_data[:, 2]

        self.indptr = np.zeros(self.n_elements + 1, dtype=int)
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

    def get_sparse_row(self, idx):
        start = self.indptr[idx]
        end   = self.indptr[idx + 1]
        return self.indices[start:end], self.csr_data[start:end]

    def similarity(self):
        self.S = np.zeros((self.n_elements, self.n_elements))

        for i in range(self.n_elements):
            items_i, ratings_i = self.get_sparse_row(i)

            for j in range(i, self.n_elements):
                if i == j:
                    self.S[i, j] = 1.0
                    continue

                items_j, ratings_j = self.get_sparse_row(j)
                common, idx_i, idx_j = np.intersect1d(items_i, items_j, return_indices=True)

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

    def predict_score_for_ranking(self, u, i):
        if u >= self.n_users or i >= self.n_items:
            return self.mu[u] if u < self.n_users else 0

        if self.mode == 'user':
            target_idx = u
            item_indices = np.where(self.Y_data[:, 1] == i)[0]
            neighbors = self.Y_data[item_indices, 0].astype(int)
            search_target = i
        else:
            target_idx = i
            user_indices = np.where(self.Y_data[:, 0] == u)[0]
            neighbors = self.Y_data[user_indices, 1].astype(int)
            search_target = u

        neighbors = neighbors[neighbors != target_idx]

        if len(neighbors) == 0:
            return self.mu[u]

        sims = self.S[target_idx, neighbors]
        top_idx   = np.argsort(sims)[::-1][:self.k]
        top_neighbors = neighbors[top_idx]
        top_sims  = np.maximum(sims[top_idx], 0)

        ratings = []
        for neighbor in top_neighbors:
            items_n, ratings_n = self.get_sparse_row(neighbor)
            idx = np.where(items_n == search_target)[0]
            ratings.append(ratings_n[idx[0]] if len(idx) > 0 else 0)

        ratings = np.array(ratings)
        denom   = np.sum(np.abs(top_sims))
        if denom == 0:
            return self.mu[u]

        return self.mu[u] + np.sum(top_sims * ratings) / denom

    def predict(self, u, i):
        pred = self.predict_score_for_ranking(u, i)
        if pred is None: return None
        return np.clip(pred, 1, 5)


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def compute_f1(p, r):
    return 2 * p * r / (p + r + 1e-8)

def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
    user_items = {}
    for row in Y:
        u, i, r, t = row[0], row[1], row[2], row[3]
        user_items.setdefault(int(u), []).append((i, r, t))

    train, valid, test = [], [], []
    for u in user_items:
        items = sorted(user_items[u], key=lambda x: x[2])
        n       = len(items)
        n_train = int(train_ratio * n)
        n_valid = int(valid_ratio * n)
        for i, r, _ in items[:n_train]: train.append([u, i, r])
        for i, r, _ in items[n_train:n_train + n_valid]: valid.append([u, i, r])
        for i, r, _ in items[n_train + n_valid:]: test.append([u, i, r])

    return np.array(train), np.array(valid), np.array(test)

def rmse(model, data):
    se, cnt = 0, 0
    for u, i, r in data:
        pred = model.predict(int(u), int(i))
        if pred is None: continue
        se  += (r - pred) ** 2
        cnt += 1
    return np.sqrt(se / cnt) if cnt > 0 else float('nan')

def evaluate_top_k(model, data, n_items, K=10, threshold=3.5, n_neg=100):
    np.random.seed(42)
    user_liked = {}
    for u, i, r in data:
        if r >= threshold:
            user_liked.setdefault(int(u), set()).add(int(i))

    precisions, recalls = [], []

    for u in user_liked:
        liked = user_liked[u]
        items_u = set(model.Y_data[model.Y_data[:, 0] == u, 1].astype(int))

        valid_liked = liked - items_u
        if not valid_liked: continue

        all_items = set(range(n_items))
        negatives = list(all_items - items_u - valid_liked)
        if len(negatives) < n_neg: continue

        negatives  = np.random.choice(negatives, n_neg, replace=False)
        candidates = list(valid_liked) + list(negatives)
        np.random.shuffle(candidates)

        preds = []
        for i in candidates:
            p = model.predict_score_for_ranking(u, i)
            if p is not None:
                p += np.random.uniform(0, 1e-6)
                preds.append((i, p))

        if not preds: continue
        preds.sort(key=lambda x: x[1], reverse=True)
        top_k = set([i for i, _ in preds[:K]])
        hits  = len(top_k & valid_liked)

        precisions.append(hits / K)
        recalls.append(hits / len(valid_liked))

    if not precisions: return 0.0, 0.0
    return np.mean(precisions), np.mean(recalls)


# ==========================================
# 3. CHẠY CHÍNH: TUNING & LEARNING CURVE
# ==========================================
if __name__ == "__main__":
    # The line below has been added to import the `drive` module.
    from google.colab import drive
    drive.mount('/content/drive')
    BASE_PATH = '/content/drive/MyDrive/movielen/'

    print("1. Đang tải và chia dữ liệu...")
    df = pd.read_csv(BASE_PATH + 'ml-100k (1)/ml-100k/u.data', sep='\t', names=['u', 'i', 'r', 't'])
    df['u'] -= 1
    df['i'] -= 1
    Y = df.values

    n_users = int(np.max(Y[:, 0])) + 1
    n_items = int(np.max(Y[:, 1])) + 1

    Y_train, Y_valid, Y_test = split_data(Y)

    # ===== HÀM TÌM BEST PARAMS CHUNG =====
    def tune_hyperparameters(mode_name):
        print(f"\n--- Bắt đầu Tuning cho {mode_name.upper()}-BASED ---")
        best_f1 = -1
        best_params = {}

        # Có thể giảm list này lại nếu bạn test chạy quá lâu
        for k in [10, 20, 30, 50]:
            for shrink in [10, 20]:
                for min_common in [3, 5]:
                    model = CFRecommender(Y_train, n_users, n_items, mode=mode_name,
                                           k=k, shrink=shrink, min_common=min_common)
                    model.fit()

                    p_val, r_val = evaluate_top_k(model, Y_valid, n_items, K=10)
                    f1_val = compute_f1(p_val, r_val)

                    print(f"[{mode_name}] k={k:2d}, shrink={shrink:2d}, min_com={min_common} "
                          f"-> P@10={p_val:.4f}, R@10={r_val:.4f}, F1={f1_val:.4f}")

                    if f1_val > best_f1:
                        best_f1 = f1_val
                        best_params = {'k': k, 'shrink': shrink, 'min_common': min_common}

        print(f"==> Best params cho {mode_name.upper()}: {best_params} (F1={best_f1:.4f})")
        return best_params

    # 2. CHẠY TUNING
    best_params_user = tune_hyperparameters('user')
    best_params_item = tune_hyperparameters('item')

    # Đánh giá Test Set với Best Params
    print("\n--- ĐÁNH GIÁ TRÊN TẬP TEST ---")

    # Test User-Based
    final_user_model = CFRecommender(Y_train, n_users, n_items, mode='user', **best_params_user)
    final_user_model.fit()
    p_test_u, r_test_u = evaluate_top_k(final_user_model, Y_test, n_items, K=10)
    print(f"USER-BASED | Test RMSE: {rmse(final_user_model, Y_test):.4f} "
          f"| P@10: {p_test_u:.4f} | R@10: {r_test_u:.4f} | F1: {compute_f1(p_test_u, r_test_u):.4f}")

    # Test Item-Based
    final_item_model = CFRecommender(Y_train, n_users, n_items, mode='item', **best_params_item)
    final_item_model.fit()
    p_test_i, r_test_i = evaluate_top_k(final_item_model, Y_test, n_items, K=10)
    print(f"ITEM-BASED | Test RMSE: {rmse(final_item_model, Y_test):.4f} "
          f"| P@10: {p_test_i:.4f} | R@10: {r_test_i:.4f} | F1: {compute_f1(p_test_i, r_test_i):.4f}")


    # 3. VẼ LEARNING CURVE VỚI BEST PARAMS
    print("\n--- VẼ LEARNING CURVE ---")
    train_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]

    user_rmses, user_p10s = [], []
    item_rmses, item_p10s = [], []

    df_full = pd.DataFrame(Y_train, columns=["u", "i", "r"])

    for frac in train_sizes:
        print(f" ► Đang xử lý mốc {frac*100:2.0f}% data...")

        Y_sub_list = []
        for u, group in df_full.groupby('u'):
            n_take = max(1, int(len(group) * frac))
            Y_sub_list.append(group.iloc[:n_take].values)
        Y_sub = np.vstack(Y_sub_list)

        # User-Based bằng best params
        model_u = CFRecommender(Y_sub, n_users, n_items, mode='user', **best_params_user)
        model_u.fit()
        user_rmses.append(rmse(model_u, Y_valid))
        p_val_u, _ = evaluate_top_k(model_u, Y_valid, n_items, K=10)
        user_p10s.append(p_val_u)

        # Item-Based bằng best params
        model_i = CFRecommender(Y_sub, n_users, n_items, mode='item', **best_params_item)
        model_i.fit()
        item_rmses.append(rmse(model_i, Y_valid))
        p_val_i, _ = evaluate_top_k(model_i, Y_valid, n_items, K=10)
        item_p10s.append(p_val_i)

    # 4. VẼ BIỂU ĐỒ
    x_axis = [int(x * 100) for x in train_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(x_axis, user_rmses, 'o-', color='tab:blue', label='User-Based')
    ax1.plot(x_axis, item_rmses, 's--', color='tab:orange', label='Item-Based')
    ax1.set_title('Validation RMSE (Càng thấp càng tốt)')
    ax1.set_xlabel('Training Data Size (%)')
    ax1.set_ylabel('RMSE')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(x_axis, user_p10s, 'o-', color='tab:green', label='User-Based')
    ax2.plot(x_axis, item_p10s, 's--', color='tab:red', label='Item-Based')
    ax2.set_title('Validation Precision@10 (Càng cao càng tốt)')
    ax2.set_xlabel('Training Data Size (%)')
    ax2.set_ylabel('P@10')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.suptitle('So sánh User-Based vs Item-Based (Đã dùng Best Params)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('learning_curve_tuned.png', dpi=200)
    plt.show()
    print("\n--- LƯU MODEL XUỐNG DRIVE ---")
    model_path = BASE_PATH + "recommender_models.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            'user_model': final_user_model,
            'item_model': final_item_model,
            'best_params_user': best_params_user,
            'best_params_item': best_params_item
        }, f)
    
    print(f"✅ Đã lưu toàn bộ Model và Best Params tại: {model_path}")