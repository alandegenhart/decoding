# Python reference

# Import
import pandas as pd
import sklearn

# Fill missing/na values in a pandas DataFrame:
pd.DataFrame.fillna()

# Train/test splits using scikit-learn
X = []  # features
y = []  # variable to predict
sklearn.model_selection.train_test_split(X, y, random_state=1)

"""
Models.
"""

# Decision tree regressor
model = sklearn.tree.DecisionTreeRegressor(random_state=1)

# Random forest -- this is a collection of trees, with the final prediction
# made by averaging the component trees.  A random forest classifier is also
# available.
model = sklearn.ensemble.RandomForestRegressor(random_state=1)
