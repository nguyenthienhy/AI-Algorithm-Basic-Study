import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def normalize_data(X):
	for i in range(X.shape[0]):
		for j in range(X.shape[1]):
			if np.isnan(X[i][j]):
				X[i][j] = 0


def readDataWithLabelEncoder(filename):
	data = pd.read_csv(filename)
	X = data.iloc[:, 1:].values
	labels = data.iloc[:, 0].values

	X, labels = shuffle(X, labels, random_state=42)
	normalize_data(X)  # chuẩn hoá dữ liệu nan

	# chuyển dữ liệu dạng chuỗi về dạng số
	le = preprocessing.LabelEncoder()
	for i in range(X.shape[1]):
		X[:, i] = le.fit_transform(X[:, i].astype(str))
	labels = le.fit_transform(labels)

	# chuẩn hoá dữ liệu
	#X = normalize(X)
	#labels = (normalize(np.array([labels])).reshape(1, -1))[0]

	df_new = pd.DataFrame(data=X)
	df_new = pd.concat([df_new, data[["group"]]], axis=1)

	return X, labels, df_new


def plot2D(X, data):
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(X)
	principalDf = pd.DataFrame(data=principalComponents, columns=['Feature 1', 'Feature 2'])
	finalDf = pd.concat([principalDf, data[["group"]]], axis=1)
	fig = plt.figure(figsize=(10, 10))
	ax = fig.add_subplot(1, 1, 1)
	ax.set_xlabel('Feature 1', fontsize=15)
	ax.set_ylabel('Feature 2', fontsize=15)
	ax.set_title('Data reduce dimention', fontsize=20)
	targets = ['C', 'P', 'R', 'I']
	colors = ['r', 'g', 'b', 'y']
	for target, color in zip(targets, colors):
		indicesToKeep = finalDf['group'] == target
		ax.scatter(finalDf.loc[indicesToKeep, 'Feature 1']
				   , finalDf.loc[indicesToKeep, 'Feature 2']
				   , c=color
				   , s=50)
	ax.legend(targets)
	ax.grid()
	plt.show()


def plot3D(df):
	my_dpi = 96
	plt.figure(figsize=(480 / my_dpi, 480 / my_dpi), dpi=my_dpi)

	# Keep the 'specie' column appart + make it numeric for coloring
	df['group'] = pd.Categorical(df['group'])
	my_color = df['group'].cat.codes
	df = df.drop('group', 1)

	# Run The PCA
	pca = PCA(n_components=3)
	pca.fit(df)

	# Store results of PCA in a data frame
	result = pd.DataFrame(pca.transform(df), columns=['PCA%i' % i for i in range(3)], index=df.index)

	# Plot initialisation
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(result['PCA0'], result['PCA1'], result['PCA2'], c=my_color, cmap="Set2_r", s=60)

	# make simple, bare axis lines through space:
	xAxisLine = ((min(result['PCA0']), max(result['PCA0'])), (0, 0), (0, 0))
	ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'r')
	yAxisLine = ((0, 0), (min(result['PCA1']), max(result['PCA1'])), (0, 0))
	ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'r')
	zAxisLine = ((0, 0), (0, 0), (min(result['PCA2']), max(result['PCA2'])))
	ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'r')

	# label the axes
	ax.set_xlabel("PC1")
	ax.set_ylabel("PC2")
	ax.set_zlabel("PC3")
	ax.set_title("PCA on personal habit data set")

	plt.show()


X_train, label_true, df = readDataWithLabelEncoder("data.csv")

print("Data train : " + str(X_train.shape[0]))

clf = KMeans(n_clusters=4, random_state=42).fit(X_train)

label_predict = clf.labels_

num_correct = 0
for i in range(len(label_true)):
	if label_true[i] == label_predict[i]:
		num_correct += 1

print("Score : " + str(num_correct / len(label_true)))
print(df)
plot2D(X_train , df)
plot3D(df)
