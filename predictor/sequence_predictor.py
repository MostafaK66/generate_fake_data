import numpy as np
import pandas as pd
from numpy import asarray
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


class UnivarientSequencePredictor:
    def __init__(self, param_grid, time_series_split_ratio):
        self.param_grid = param_grid
        self.time_series_split_ratio = time_series_split_ratio

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
        RF = GridSearchCV(
            estimator=RandomForestRegressor(random_state=123),
            param_grid=self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.time_series_split_ratio),
        )
        RF.fit(trainX, trainy)
        best_model = RF.best_estimator_
        yhat = best_model.predict([testX])
        regressor_name = best_model.__class__.__name__

        return yhat[0], {regressor_name: RF.best_params_}

    def walk_forward_validation(self, train, test):
        """
        Conducts walk forward validation, fitting the model and making predictions step-by-step.

        Args:
            train (array-like): The training dataset.
            test (array-like): The test dataset.

        Returns:
            float: The Mean Absolute Error between predictions and actuals.
            DataFrame: A DataFrame containing columns 'Actual' and 'Predicted' for test data.
        """
        predictions = list()
        actuals = list()
        best_params_dict = {}
        history = [x for x in train]

        for i in range(len(test)):
            testX, testy = test[i, :-1], test[i, -1]
            yhat, best_params = self.univarient_predictor(history, testX)
            yhat_rounded = round(yhat)
            predictions.append(yhat_rounded)
            actuals.append(testy)
            best_params_dict.update(best_params)
            history.append(test[i])

        mean_absolute_error_value = mean_absolute_error(actuals, predictions)

        results_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})

        return mean_absolute_error_value, results_df, best_params_dict
