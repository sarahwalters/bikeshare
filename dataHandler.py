import csv
import numpy as np
import sklearn.cross_validation as cv
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing


def readData(featCols):
	''' makes feature/label matrices from training data csv '''

	# read csv and make numpy matrix
	with open('train.csv', 'rb') as csvfile:
	    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
	    rows = []
	    for i, row in enumerate(reader):
	    	if i == 0: # don't split label row -> replace datetime with time
	    		row[0] = "time"
	    	else:
	    		# row[0] is just hour, not timestamp
		    	datetimeArray = row[0].split(" ")
		    	time = datetimeArray[1]
		    	hourStr = str(time)[0:2]
		    	row[0] = int(hourStr)
		    	rows.append(row)
	rowMatrix = np.array(rows, dtype=np.float64)

	# separate features (X) from labels (Y)
	(r, c) = rowMatrix.shape

	X = rowMatrix[:,featCols] 	# features
	y_cas = rowMatrix[:,c-3]	# casual uses
	y_reg = rowMatrix[:,c-2]	# registered uses
	y = rowMatrix[:,c-1]		# total uses

	return [X, y_cas, y_reg, y]


def learn(feats):
	[X, yc, yr, y] = readData(feats)
	[X_tr, X_te, yr_tr, yr_te] = cv.train_test_split(X, yr, test_size=0.1)

	minmaxscaler = preprocessing.MinMaxScaler()
	X_tr_scaled = minmaxscaler.fit_transform(X_tr)
	X_te_scaled = minmaxscaler.transform(X_te)

	# fit clf
	clf = LinearRegression(fit_intercept=True)
	clf.fit(X_tr_scaled, yr_tr)

	# evaluate
	return clf.score(X_te_scaled, yr_te)


def chooseFeatSet():
	f_all = [0, 1, 2, 3, 4, 5, 6, 7, 8]
	f_new = [0, 2, 3, 4, 5, 6, 7, 8]
	compareFeatSets(f_all, f_new)

	# 0: 100/0	 	3: 55/45	6: 45/55
	# 1: 76/24 		4: 52/48	7: 87/13
	# 2: 51/49 		5: 49/51	8: 46/54

def compareFeatSets(f1, f2):
	score1 = 0
	score2 = 0
	for i in range(100):
		s1 = learn(f1)
		s2 = learn(f2)
		if s1>s2:
			print s1
			print s2
			print '---'
			score1 = score1+1
		else:
			score2 = score2+1
	print '---'
	print score1
	print score2