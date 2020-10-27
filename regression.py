import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score, mean_squared_error

class MainRegressor(object):

    def __init__(self, dataset, target):
        self.dataset = dataset
        self.target = target
        self.features = [i for i in list(dataset.columns) if i!= target]

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None

    def splitting(self):
        X = dataset[self.features]
        y = dataset[self.target]

        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y, 
                                                            test_size = 0.33, 
                                                            random_state = 101
                                                            )

    

    def linear_regression(self):
        self.splitting()

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def linear_regression_score(self):
        pred = self.model.predict(X_test)
        return (mean_squared_error(y_test, pred), r2_score(y_test, pred))

    
    def logistic_regression(self):
        self.splitting()

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def logistic_regression_score(self):
        pred = self.model.predict(X_test)
        return (mean_squared_error(y_test, pred), r2_score(y_test, pred))

    def decisionTree(self):
        self.splitting()

        self.model = DecisionTreeRegressor()
        self.model.fit(self.X_train, self.y_train)

    def decisionTree_score(self):
        pred = self.model.predict(X_test)
        return (mean_squared_error(y_test, pred), r2_score(y_test, pred))

    def randomForest(self):
        self.splitting()

        self.model = RandomForestRegressor()
        self.model.fit(self.X_train, self.y_train)

    def randomForest_score(self):
        pred = self.model.predict(X_test)
        return (mean_squared_error(y_test, pred), r2_score(y_test, pred))