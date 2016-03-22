import numpy as np
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score

'''evaluating classifier performance overview 4.8.1'''
iris = datasets.load_iris() #loading iris data from sklearn's library

X = iris.data[0:150, 1:3]
y = iris.target[0:150]

kf = KFold(150, n_folds=5, shuffle = True) #defining a list where each list has (1. list of row numbers for Training set, 2. list of row numbers for testing set)

avg = []
f_one = []
cnt=0
for train, test in kf:
    X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
    svc = svm.SVC(kernel='linear', C=1) #The > C, the wider the margin between groups
    svc.fit(X_train, y_train)
    g = svc.predict(X_test)
    yhat = (svc.predict(X_test))
    avg.append(svc.score(X_train, y_train))
    print test
    #print f_one
    print f1_score(y_test, yhat)
    f_one.append(f1_score(y_test, yhat))

print ""
print "Mean accuracy score on test data:"
print avg
print ""
print "Average cross-valiadtion SVC score:"
print np.average(avg)
print ""
print "Standard deviation of cross-valiadtion SVC scores:"
print np.std(avg)
print ""
print "f1 score array"
print f_one

#print f1_score(y_test, f_one[0])
