import pandas as pd
import numpy as np
from sklearn.utils import shuffle

df = pd.read_csv("data.csv")
df2 = pd.read_csv("data_bbc.csv")

X_1 = df["content"].values
y_1 = df["category"].values
X_2 = df2["content"].values
y_2 = df2["category"].values

X = np.concatenate((X_1 , X_2) , axis=0)
Y = np.concatenate((y_1 , y_2) , axis=0)

X , Y = shuffle(X , Y , random_state=42)

df = pd.DataFrame()
df['content'] = X
df['category'] = Y
df.to_csv("data_total.csv", encoding="utf-8")
