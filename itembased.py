import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle

class CF(object):
    def __init__(self, Y_data, k, dist_func=cosine_similarity, uuCF=1):
        self.uuCF = uuCF
        self.Y_data = Y_data if uuCF else Y_data[:, [1, 0, 2]]
        self.k = k
        self.dist_func = dist_func
        self.Ybar_data = None
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def normalize_Y(self):
        users = self.Y_data[:, 0]
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users,))
        for n in range(self.n_users):
            ids = np.where(users == n)[0].astype(np.int32)
            if len(ids) == 0:
                self.mu[n] = 0
                continue
            ratings = self.Y_data[ids, 2]
            m = np.mean(ratings)
            self.mu[n] = m
            self.Ybar_data[ids, 2] = ratings - self.mu[n]

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    def fit(self):
        self.normalize_Y()
        self.similarity()

    def __pred(self, u, i, normalized=1):
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        sim = self.S[u, users_rated_i]
        a = np.argsort(sim)[-self.k:]
        nearest_s = sim[a]
        r = self.Ybar[i, users_rated_i[a]]
        res = (r * nearest_s)[0] / (np.abs(nearest_s).sum() + 1e-8)
        return res if normalized else res + self.mu[u]

    def recommend(self, u):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)
        return recommended_items


# ── MAIN ──────────────────────────────────────────────────
if __name__ == "__main__":
    ratings = pd.read_csv('dataset/ml-100k (1)/ml-100k/u.data', sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    Y_data = ratings[['user_id', 'item_id', 'rating']].values

    rs = CF(Y_data, k=10, uuCF=0)
    rs.fit()
    print("Train mô hình thành công!")

    with open("cf_model.pkl", "wb") as f:
        pickle.dump(rs, f)
    print("Đã lưu model vào cf_model.pkl")

    print(f"Top 10 gợi ý: {rs.recommend(0)[:10]}")