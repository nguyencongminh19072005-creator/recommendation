import pandas as pd
import numpy as np

class UserBasedCF:
    def __init__(self, Y_data, n_users, n_items, k=50):
        self.Y_data = Y_data
        self.k = k
        self.n_users = n_users
        self.n_items = n_items

    def normalize(self):
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        users = self.Y_data[:, 0]

        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            ratings = self.Y_data[ids, 2]

            if len(ratings) > 0:
                m = np.mean(ratings)
            else:
                m = 0

            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        rows = self.Ybar_data[:, 0].astype(int)
        cols = self.Ybar_data[:, 1].astype(int)
        data = self.Ybar_data[:, 2]

        self.indptr = np.zeros(self.n_users + 1, dtype=int)
        for r in rows:
            self.indptr[r + 1] += 1
        self.indptr = np.cumsum(self.indptr)

        self.indices = np.zeros(len(cols), dtype=int)
        self.csr_data = np.zeros(len(data), dtype=float)

        tracker = self.indptr[:-1].copy()
        for i in range(len(data)):
            r = rows[i]
            dest = tracker[r]
            self.indices[dest] = cols[i]
            self.csr_data[dest] = data[i]
            tracker[r] += 1

    def get_sparse_row(self, u):
        start = self.indptr[u]
        end = self.indptr[u + 1]
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
                comm_items, c_i, c_j = np.intersect1d(items_i, items_j, return_indices=True)

                n_common = len(comm_items)

                if n_common < 2:
                    self.S[i, j] = self.S[j, i] = 0.0
                    continue

                r_i = ratings_i[c_i]
                r_j = ratings_j[c_j]

                mean_i = np.mean(r_i)
                mean_j = np.mean(r_j)

                diff_i = r_i - mean_i
                diff_j = r_j - mean_j

                numerator = np.sum(diff_i * diff_j)
                denominator = np.sqrt(np.sum(diff_i**2)) * np.sqrt(np.sum(diff_j**2))

                if denominator == 0:
                    sim = 0.0
                else:
                    sim = numerator / denominator

                # 🔥 shrink (giữ lại)
                shrink = n_common / (n_common + 10)
                sim = sim * shrink

                self.S[i, j] = self.S[j, i] = sim

    def fit(self):
        self.normalize()
        self.similarity()

    def predict(self, u, i):
        item_indices = np.where(self.Y_data[:, 1] == i)[0]

        if len(item_indices) == 0:
            return 3.0 if self.mu[u] == 0 else np.clip(self.mu[u], 1, 5)

        users_rated_i = self.Y_data[item_indices, 0].astype(int)
        sims = self.S[u, users_rated_i]

        sorted_idx = np.argsort(sims)[::-1]
        top_k_idx = sorted_idx[:self.k]

        top_users = users_rated_i[top_k_idx]
        top_sims = sims[top_k_idx]

        # 🔥 chỉ bỏ similarity âm
        top_sims = np.maximum(top_sims, 0)

        top_normalized_ratings = []
        for user in top_users:
            items_u, ratings_u = self.get_sparse_row(user)
            idx = np.where(items_u == i)[0]
            if len(idx) > 0:
                top_normalized_ratings.append(ratings_u[idx[0]])
            else:
                top_normalized_ratings.append(0)

        top_normalized_ratings = np.array(top_normalized_ratings)
        denominator = np.sum(np.abs(top_sims))

        if denominator == 0:
            return 3.0 if self.mu[u] == 0 else np.clip(self.mu[u], 1, 5)

        numerator = np.sum(top_sims * top_normalized_ratings)
        predicted_score = self.mu[u] + numerator / denominator

        return np.clip(predicted_score, 1, 5)

    def recommend(self, u, n_recommendations=5):
        items_u, _ = self.get_sparse_row(u)
        all_items = np.arange(self.n_items)
        unrated_items = np.setdiff1d(all_items, items_u)

        predictions = []
        for i in unrated_items:
            pred = self.predict(u, i)
            predictions.append((i, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]


def split_data(Y, train_ratio=0.7, valid_ratio=0.1):
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
        for i, r in items[n_train:n_train+n_valid]:
            valid.append([u, i, r])
        for i, r in items[n_train+n_valid:]:
            test.append([u, i, r])

    return np.array(train), np.array(valid), np.array(test)


def rmse(model, data):
    se = 0
    for u, i, r in data:
        pred = model.predict(int(u), int(i))
        se += (r - pred) ** 2
    return np.sqrt(se / len(data))


def evaluate_top_k(model, data, n_items, K=10, threshold=4, n_neg=300):
    user_liked_test = {}
    train_users = set(model.Y_data[:, 0].astype(int))

    for u, i, r in data:
        u = int(u)
        i = int(i)
        if r >= threshold and u in train_users:
            user_liked_test.setdefault(u, set()).add(i)

    precisions, recalls = [], []

    for u in user_liked_test:
        liked_items = user_liked_test[u]

        items_u_train, _ = model.get_sparse_row(u)
        items_u_train = set(items_u_train)

        valid_liked = liked_items - items_u_train
        if len(valid_liked) == 0:
            continue

        all_items = set(range(n_items))
        unrated_items = list(all_items - items_u_train - valid_liked)

        if len(unrated_items) < n_neg:
            continue

        negatives = np.random.choice(unrated_items, n_neg, replace=False)
        candidates = list(valid_liked) + list(negatives)

        predictions = []
        for i in candidates:
            pred = model.predict(u, i)
            predictions.append((i, pred))

        predictions.sort(key=lambda x: x[1], reverse=True)
        top_k_items = set([item for item, _ in predictions[:K]])

        hits = len(top_k_items & valid_liked)

        precisions.append(hits / K)
        recalls.append(hits / len(valid_liked))

    if len(precisions) == 0:
        return 0.0, 0.0

    return np.mean(precisions), np.mean(recalls)


header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dataset/ml-100k (1)/ml-100k/u.data', sep='\t', names=header)
df = df.drop('timestamp', axis=1)

df['user_id'] -= 1
df['item_id'] -= 1

Y_raw = df.values

n_users = int(np.max(Y_raw[:, 0])) + 1
n_items = int(np.max(Y_raw[:, 1])) + 1

np.random.seed(42)
Y_train, Y_valid, Y_test = split_data(Y_raw)

model = UserBasedCF(Y_train, n_users, n_items, k=50)
model.fit()

print("RMSE Train:", np.round(rmse(model, Y_train), 4))
print("RMSE Valid:", np.round(rmse(model, Y_valid), 4))
print("RMSE Test :", np.round(rmse(model, Y_test), 4))

for k in [5, 10, 15]:
    p_valid, r_valid = evaluate_top_k(model, Y_valid, n_items, K=k)
    p_test, r_test = evaluate_top_k(model, Y_test, n_items, K=k)

    print("\nK =", k)
    print("Precision Valid:", np.round(p_valid * 100, 2), "%")
    print("Precision Test :", np.round(p_test * 100, 2), "%")
    print("Recall Valid   :", np.round(r_valid * 100, 2), "%")
    print("Recall Test    :", np.round(r_test * 100, 2), "%")