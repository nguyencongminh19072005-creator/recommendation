import pandas as pd
import numpy as np
from recommendation import split_data

# 1. Load data
print("=== 1. Load Data ===")
df = pd.read_csv(r'd:\BTL_AI\dataset\ml-100k (1)\ml-100k\u.data', sep='\t', names=['u', 'i', 'r', 't'])
Y = df.values

# 2. Phân bố số tương tác trên user
print("\n=== 2. Interaction Distribution per User ===")
counts = df.groupby('u').size()
users_1 = (counts == 1).sum()
users_2 = (counts == 2).sum()
users_3_plus = (counts >= 3).sum()

print(f"Total users: {len(counts)}")
print(f"Users with 1 interaction: {users_1} ({users_1/len(counts)*100:.2f}%)")
print(f"Users with 2 interactions: {users_2} ({users_2/len(counts)*100:.2f}%)")
print(f"Users with >= 3 interactions: {users_3_plus} ({users_3_plus/len(counts)*100:.2f}%)")
print(f"Minimum interactions for a user: {counts.min()}")

# 3. Sanity checks sau split
print("\n=== 3. Running split_data and Sanity Checks ===")
train, valid, test = split_data(Y)

train_df = pd.DataFrame(train, columns=['u', 'i', 'r'])
valid_df = pd.DataFrame(valid, columns=['u', 'i', 'r'])
test_df = pd.DataFrame(test, columns=['u', 'i', 'r'])

train_users = set(train_df['u'])
valid_users = set(valid_df['u'])
test_users = set(test_df['u'])

print("Subset checks:")
is_valid_subset = valid_users.issubset(train_users)
is_test_subset = test_users.issubset(train_users)
print(f"  valid.user is subset of train.user: {is_valid_subset}")
print(f"  test.user is subset of train.user: {is_test_subset}")

print("\nTotal records check:")
total_records = len(train) + len(valid) + len(test)
print(f"  len(train) + len(valid) + len(test) = {total_records}")
print(f"  len(Y) = {len(Y)}")
print(f"  Equal: {total_records == len(Y)}")

print("\nEmpty test users check:")
users_no_test = len(train_users - test_users)
print(f"  Users present in train but absent in test: {users_no_test} / {len(train_users)}")

# 4. In thử chi tiết nếu có lỗi
if not is_valid_subset:
    print("\n[ERROR] Users in valid but NOT in train:", valid_users - train_users)
if not is_test_subset:
    print("\n[ERROR] Users in test but NOT in train:", test_users - train_users)
