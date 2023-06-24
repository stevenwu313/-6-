import numpy as np
import pandas as pd
import sklearn.tree as tree
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv("C:/Users/1/Desktop/recipes_train.csv")
test_df = pd.read_csv("C:/Users/1/Desktop/recipes_test.csv")

x = train_df.drop("cuisine", axis = 1)   #分离,不能包含标签列
y = train_df["cuisine"]    #留下标签列
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 40)

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
score = clf.score(X_train, Y_train)
score
score1 = clf.score(X_test, Y_test)
score1
result = clf.predict(test_df.values)
outcome = pd.DataFrame()
outcome["id"] = test_df["id"]
outcome["cuisine"] = result
outcome.to_csv("Desktop/submission1.csv", index = False)
