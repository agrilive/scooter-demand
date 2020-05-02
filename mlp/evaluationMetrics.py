import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

class PerformanceMetrics:
    """
    Defines methods to evaluate the model
    Parameters
    ----------
    y_actual : array-like, shape = [n_samples]
            Observed values from the training samples
    y_predicted : array-like, shape = [n_samples]
            Predicted values from the model
    """
    def rmsle(self, y_actual, y_predicted):
        error = np.sqrt(mean_squared_log_error(y_actual, y_predicted))
        return error
