import numpy as np
from sklearn.ensemble import RandomForestRegressor
from numpy import asarray


class UnivarientSequencePredictor:

    def __init__(self):
        pass

    def univarient_predictor(self, train, testX):
        """
        Fits a Random Forest Regressor on the training data and makes a prediction.

        Args:
            train (array-like): The training data. Last column should be the target variable.
            testX (array-like): The test input features.

        Returns:
            float: Predicted value for the testX.
        """
        train = asarray(train)
        trainX, trainy = train[:, :-1], train[:, -1]
        model = RandomForestRegressor(n_estimators=1000)
        model.fit(trainX, trainy)
        yhat = model.predict([testX])
        return yhat[0]
