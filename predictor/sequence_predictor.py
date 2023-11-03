import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
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

    def walk_forward_validation(self, data, train, test):
        """
        Conducts walk forward validation, fitting the model and making predictions step-by-step.

        Args:
            data (array-like): The complete dataset.
            train (array-like): The training dataset.
            test (array-like): The test dataset.

        Returns:
            float: The Mean Absolute Error between predictions and actuals.
            DataFrame: A DataFrame containing columns 'Actual' and 'Predicted' for test data.
        """
        predictions = list()
        actuals = list()
        history = [x for x in train]

        for i in range(len(test)):
            testX, testy = test[i, :-1], test[i, -1]
            yhat = self.univarient_predictor(history, testX)
            predictions.append(yhat)
            actuals.append(testy)
            history.append(test[i])

        mean_absolute_error_value = mean_absolute_error(actuals, predictions)

        results_df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})

        return mean_absolute_error_value, results_df


