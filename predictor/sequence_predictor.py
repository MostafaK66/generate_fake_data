import numpy as np
import pandas as pd
from numpy import asarray
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor


class UnivarientSequencePredictor:
    def __init__(self, param_grid, time_series_split_ratio):
        self.param_grid = param_grid
        self.time_series_split_ratio = time_series_split_ratio

    def univarient_predictor(self, train, testX):
        train = np.asarray(train)
        trainX, trainy = train[:, :-1], train[:, -1]
        estimators = [
            ("xgb", XGBRegressor(device="cpu", verbosity=1, random_state=123))
        ]
        stack = StackingRegressor(
            estimators=estimators, final_estimator=SGDRegressor(max_iter=1000)
        )

        stack_grid_search = GridSearchCV(
            estimator=stack,
            param_grid=self.param_grid,
            cv=TimeSeriesSplit(n_splits=self.time_series_split_ratio),
            n_jobs=1,
        )
        stack_grid_search.fit(trainX, trainy)
        best_model = stack_grid_search.best_estimator_
        yhat = best_model.predict([testX])
        regressor_name = "StackingRegressor"

        return yhat[0], {regressor_name: stack_grid_search.best_params_}

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
