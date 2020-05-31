from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn import metrics
import pandas as pd

def readDataWithLabelEncoder(filename):
	data = pd.read_csv(filename)
	X = data.iloc[: , 1 : ].values
	y = data.iloc[: , 0].values
	X , y = shuffle(X , y , random_state = 42)

	X_train_and_val , X_test , y_train_and_val , y_test = train_test_split(X, y , test_size = 0.2 , random_state = 42)

	# xử lý miss values
	imp_mean = SimpleImputer(strategy='most_frequent')
	imp_mean.fit(X_train_and_val)
	X_train_and_val = imp_mean.transform(X_train_and_val)
	
	le = preprocessing.LabelEncoder()
	for i in range(X_train_and_val.shape[1]):
		X_train_and_val[: , i] = le.fit_transform(X_train_and_val[: , i].astype(str))

	for i in range(X_test.shape[1]):
		X_test[: , i] = le.fit_transform(X_test[: , i].astype(str))

	return X_train_and_val , X_test , y_train_and_val , y_test


X_train_and_val , X_test , y_train_and_val , y_test = readDataWithLabelEncoder("mushrooms.csv")

X_train , X_val , y_train , y_val = train_test_split(X_train_and_val, y_train_and_val , 
														test_size = 0.2 , random_state = 42)

print("Train example : " + str(X_train.shape[0]))
print("Validation example : " + str(X_val.shape[0]))
print("Test example : " + str(X_test.shape[0]))

clf = RandomForestClassifier(max_depth = 3 , max_features="sqrt")
clf.fit(X_train, y_train)

print("Training accuracy : ", clf.score(X_train , y_train))
print("Validation accuracy: " , metrics.accuracy_score(clf.predict(X_val) , y_val))
print("Test accuracy: " , metrics.accuracy_score(clf.predict(X_test) , y_test))
