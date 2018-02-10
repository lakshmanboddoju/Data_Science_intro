from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier




# Decision Tree Classifier

clf1 = tree.DecisionTreeClassifier()

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

clf1.fit(X, Y)

prediction1 = clf1.predict([[190, 70, 43]])

print("\nDecision Tree Classifier: \t", prediction1)
# Naive Bayes classifier

clf2 = GaussianNB()

clf2.fit(X,Y)

prediction2 = clf2.predict([[190, 70, 43]])

print("Navie Bayes Classifier: \t", prediction2)

# K Nearest Neighbors

clf3 = KNeighborsClassifier(n_neighbors=3)

clf3.fit(X,Y)

prediction3 = clf3.predict([[190, 70, 43]])

print("K Nearest Neighbors Classifier: \t", prediction3)

# Logistic Regression

clf4 = LogisticRegression()

clf4.fit(X, Y)

prediction4 = clf4.predict([[190, 70, 43]])

print("Logistic Regression Classifier: \t", prediction4)

# Random Forest

clf5 = RandomForestClassifier(n_estimators=2)

clf5.fit(X, Y)

prediction5 = clf5.predict([[190, 70, 43]])

print("Random Forest Classifier: \t", prediction5)



