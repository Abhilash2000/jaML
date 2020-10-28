import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

class MainClassifiers(object):

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
        return accuracy_score(y_test, pred)

    
    def logistic_regression(self):
        self.splitting()

        if self.hypertune == 'No':
            self.model = LogisticRegression()
            self.model.fit(self.X_train, self.y_train)
        else:
            c_space = np.logspace(-5, 8, 15) 
            param_grid = {'C': c_space} 

            if self.kind == 'GridSearchCV':
                self.model = GridSearchCV(LogisticRegression(), 
                                          param_grid, 
                                          cv = 5, 
                                          verbose = 0
                                          )

                self.model.fit(self.X_train, self.y_train)
            
            else:
                self.model = RandomizedSearchCV(LogisticRegression(), 
                                                param_grid, 
                                                cv = 5, 
                                                verbose = 0
                                               )

                self.model.fit(self.X_train, self.y_train)

    def logistic_regression_score(self):
        pred = self.model.predict(X_test)
        return accuracy_score(y_test, pred)


    def decision_tree_classifier(self):
        self.splitting()

        if hypertune == 'No':
            self.model = DecisionTreeClassifier()
            self.model.fit(self.X_train, self.y_train)
        else:
            param_grid = {"max_depth": [3, None], 
                          "max_features": randint(1, 9), 
                          "min_samples_leaf": randint(1, 9), 
                          "criterion": ["gini", "entropy"]
                          } 

            if self.kind == 'GridSearchCV':
                self.model = GridSearchCV(DecisionTreeClassifier(), 
                                          param_grid, 
                                          cv = 5, 
                                          verbose = 0
                                          )

                self.model.fit(self.X_train, self.y_train)
            
            else:
                self.model = RandomizedSearchCV(DecisionTreeClassifier(), 
                                                param_grid, 
                                                cv = 5, 
                                                verbose = 0
                                                )

                self.model.fit(self.X_train, self.y_train)


    def decision_tree_classifier_score(self):
        pred = self.model.predict(X_test)
        return accuracy_score(y_test, pred)

    
    def random_forest_classifier(self):
        self.splitting()

        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)

    def random_forest_classifier_score(self):
        pred = self.model.predict(X_test)
        return accuracy_score(y_test, pred)

        


    