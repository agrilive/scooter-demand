from sklearn.model_selection import TimeSeriesSplit

class TimeSSplit:
    """
    Split dataset into training and test sets with sklearn's TimeSeriesSplit
    Parameters
    ----------
    X, y: array-like
        X: indepedent variables
        y: dependent variables
    """
    def split(self, X, y):
        tscv = TimeSeriesSplit(n_splits=5)
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
        return X_train, X_test, y_train, y_test