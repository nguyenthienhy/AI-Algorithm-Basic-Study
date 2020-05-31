import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score


def normalize_data(X):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if np.isnan(X[i][j]):
				X[i][j] = -1.0

def readDataWithLabelEncoder(filename):

	data = pd.read_csv(filename)
	X = data.iloc[: , 0 : ].values

	X = shuffle(X, random_state=42)
	normalize_data(X[: , 1 : ])  # chuẩn hoá dữ liệu nan

	# chuyển dữ liệu dạng chuỗi về dạng số
	le = preprocessing.LabelEncoder()
	X[:, 0] = le.fit_transform(X[:, 0].astype(str))

	# chuẩn hoá dữ liệu theo tiêu chuẩn dùng max
	X = normalize(X , norm='max')

	df_new = pd.DataFrame(data=X)

	return X, df_new

def silhouette_analysis(X_train):
	for k in list(range(2 , 11)):

		km = KMeans(n_clusters = k)
		labels = km.fit_predict(X_train)

		# Tính điểm silhouette cho mỗi điểm dữ liệu
		silhouette_vals = silhouette_samples(X_train , labels)
		assert (len(silhouette_vals) == X_train.shape[0])
		print(silhouette_vals)

		silhouette_avg = silhouette_score(X_train, labels)

		print("For n_clusters = ", k,
			  "The average silhouette_score is : ", silhouette_avg)

X_train, df = readDataWithLabelEncoder("data.csv")
print("Data Frame : ")
print(X_train)
silhouette_analysis(X_train)
