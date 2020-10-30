import pandas as pd
import numpy as np
import random as rd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error

class MainRegressors(object):

    def __init__(self, dataset, target, hypertune, kind):
        self.dataset = dataset
        self.target = target
        self.features = [i for i in list(dataset.columns) if i!= target]

        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        self.model = None
        self.hypertune = hypertune
        self.kind = kind

    def splitting(self):
        X = self.dataset[self.features]
        y = self.dataset[self.target]

        self.X_train, self.y_train, self.X_test, self.y_test = train_test_split(X, y, 
                                                            test_size = 0.33, 
                                                            random_state = 101
                                                            )

    def simple_linear_regression(self, feat):
        X = np.array(self.dataset[feat])
        y = np.array(self.dataset[self.target])

        self.model = LinearRegression()
        self.model.fit(X, y)


    def simple_linear_regression_score(self):
        pred = self.model.predict(self.X_test)
        return accuracy_score(self.y_test, pred)


    def multi_linear_regression(self):
        self.splitting()

        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

    def multi_linear_regression_score(self):
        pred = self.model.predict(self.X_test)
        return (mean_squared_error(self.y_test, pred), r2_score(self.y_test, pred))

    
    def logistic_regression(self):
        self.splitting()

        if self.hypertune == 'No':
            self.model = LogisticRegression()
            self.model.fit(self.X_train, self.y_train)
        else:
            c_space = np.logspace(-5,8,15)
            param_grid = {'C': c_space}

            if self.kind == 'GridSearchCV':
                self.model = GridSearchCV(LogisticRegression(),param_grid, cv = 5, verbose = 0)
                self.model.fit(self.X_train, self.y_train)
            else:
                self.model = RandomizedSearchCV(LogisticRegression(),param_grid, cv = 5, verbose = 0)
                self.model.fit(self.X_train, self.y_train)

    def logistic_regression_score(self):
        pred = self.model.predict(self.X_test)
        return (mean_squared_error(self.y_test, pred), r2_score(self.y_test, pred))

    def decision_tree(self):
        self.splitting()

        if self.hypertune == 'No':
            self.model = DecisionTreeRegressor()
            self.model.fit(self.X_train, self.y_train)
        else:
            param_grid = {'criterion': ["mse", "mae"],
                          'min_samples_split': [10, 20, 40],
                          'max_depth': [2, 6, 8],
                          'min_samples_leaf': [20, 40, 100],
                          'max_leaf_nodes': [5, 20, 100],
                            }

            if self.kind == 'GridSearchCV':
                self.model = GridSearchCV(DecisionTreeRegressor(), 
                                          param_grid, 
                                          cv = 5, 
                                          verbose = 0
                                          )

                self.model.fit(self.X_train, self.y_train)
            
            else:
                self.model = RandomizedSearchCV(DecisionTreeRegressor(), 
                                                param_grid, 
                                                cv = 5, 
                                                verbose = 0
                                                )

                self.model.fit(self.X_train, self.y_train)

    def decision_tree_score(self):
        pred = self.model.predict(self.X_test)
        return (mean_squared_error(self.y_test, pred), r2_score(self.y_test, pred))

    def random_forest(self):
        self.splitting()

        if self.hypertune == 'No':
            self.model = RandomForestRegressor()
            self.model.fit(self.X_train, self.y_train)
        else:
            param_grid = {'max_depth': [3, None],
                          'max_features': rd.randint(1, self.X_train.shape[1]),
                          'min_samples_split': rd.randint(2, 11),
                          'bootstrap': [True, False],
                          'n_estimators': rd.randint(100, 500)
                        }
            if self.kind == 'GridSearchCV':
                self.model = GridSearchCV(RandomForestRegressor(), 
                                          param_grid, 
                                          cv = 5, 
                                          verbose = 0
                                          )

                self.model.fit(self.X_train, self.y_train)
            
            else:
                self.model = RandomizedSearchCV(RandomForestRegressor(), 
                                                param_grid, 
                                                cv = 5, 
                                                verbose = 0
                                                )

                self.model.fit(self.X_train, self.y_train)

    def random_forest_score(self):
        pred = self.model.predict(self.X_test)
        return (mean_squared_error(self.y_test, pred), r2_score(self.y_test, pred))