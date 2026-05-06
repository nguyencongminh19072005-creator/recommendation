import pandas as pd
import matplotlib.pyplot as plt

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('dataset/ml-100k (1)/ml-100k/u.data', sep='\t', names=column_names)

plt.hist(df['rating'], bins=5)
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Movies') 
plt.grid(True)
plt.show()