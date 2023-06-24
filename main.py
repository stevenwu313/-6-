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

train_df.shape
train_df.head(20)

food_count = {}
for _,group in train_df.groupby("cuisine"):
    location = group["cuisine"].head(1).item()
    food_count[location] = {}
    for col in group.columns:
        if col not in ["id", "cuisine"]:
            food_count[location][col] = group[col].sum()
food_count.keys()
train_df.info()
train_df.describe()