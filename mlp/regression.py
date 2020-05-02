from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import numpy as np

from evaluationMetrics import PerformanceMetrics

metrics = PerformanceMetrics()
rmsle_scorer = make_scorer(metrics.rmsle, greater_is_better=False)

class RegressionModel:
    """
    Fit model with sklearn's GridSearchCV with RMSLE scoring. Predicts model and returns RMSLE score on test set
    Parameters
    ----------
    model_type : sklearn supervised learning models
            Model to train on training set
    model_params : dictionary with parameter variable and values
            Parameters for GridSearchCV
    """
    def __init__(self, model_type, model_params):
        self.model_type = model_type
        self.model_params = model_params

    def grid_fit(self, X_train, y_train):
        self.grid_model = GridSearchCV(self.model_type,
                                self.model_params,
                                scoring=rmsle_scorer,
                                cv=5)
        self.grid_model.fit(X_train, y_train)
        return self

    def predict(self, X_test, y_test):
        self.grid_model.predict(X_test)
        print('Best params: {0}'.format(self.grid_model.best_params_))
        print('Test set RMSLE score: {0}'.format(metrics.rmsle(y_test, self.grid_model.best_estimator_.predict(X_test))))
        return self.grid_model.best_estimator_

    def best_model(self, X, y):
        best_pred = self.grid_model.best_estimator_.predict(X)
        pred = np.round(np.exp(best_pred) - 1)
        return pred
