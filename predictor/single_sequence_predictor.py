import os

import joblib
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from xgboost import XGBRegressor


class SingleUnivarientSequencePredictor:
    def __init__(
        self, param_distributions, time_series_split_ratio, n_iter, optimize_model=False
    ):
        self.param_distributions = param_distributions
        self.time_series_split_ratio = time_series_split_ratio
        self.n_iter = n_iter
        self.optimize_model = optimize_model

    def load_model(self, model_path):
        return joblib.load(model_path)

    def univarient_predictor(self, train, testX, model_name=None):
        train = np.asarray(train)
        trainX, trainy = train[:, :-1], train[:, -1]

        if self.optimize_model:
            xgboosting = RandomizedSearchCV(
                estimator=XGBRegressor(),
                param_distributions=self.param_distributions,
                cv=TimeSeriesSplit(n_splits=self.time_series_split_ratio),
                random_state=123,
                n_jobs=1,
            )
            xgboosting.fit(trainX, trainy)
            best_model = xgboosting.best_estimator_
        else:
            model_path = os.path.join(
                os.getcwd(), "best_models", f"{model_name}.joblib"
            )
            best_model = self.load_model(model_path)

        yhat = best_model.predict([testX])
        regressor_name = "XGBoostRegressor" if self.optimize_model else model_name

        return (
            yhat[0],
            {regressor_name: getattr(best_model, "best_params_", {})},
            best_model,
        )

    def save_best_model(self, best_model, model_name):
        model_dir = os.path.join(os.getcwd(), "best_models")
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        joblib.dump(best_model, model_path)
        print(f"Saved best model to {model_path}")

    def walk_forward_validation(self, train, test, model_name=None):
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
            yhat, best_params, model = self.univarient_predictor(
                history, testX, model_name=model_name
            )
            yhat_rounded = round(yhat)
            predictions.append(yhat_rounded)
            actuals.append(testy)
            best_params_dict.update(best_params)
            history.append(test[i])
            best_model = model

        mean_absolute_error_value = mean_absolute_error(actuals, predictions)

        results_df = pd.DataFrame({"Actual": actuals, "Predicted": predictions})

        return mean_absolute_error_value, results_df, best_params_dict, best_model
