from pathlib import Path

script_dir = Path(__file__).resolve().parent
filename = (
    script_dir / ".." / "mocked_up" / "ada_output" / "ada_df_generator_output.csv"
)
N_IN = 4
N_OUT = 1
SPLIT_RATIO = 0.80
RF_PARAM_GRID = {
    "xgb__n_estimators": [100],
    "xgb__booster": ["gbtree", "dart"],
    "xgb__learning_rate": [0.01, 0.1],
    "xgb__max_depth": [4, 6, 20],
}
TIME_SERIES_SPLIT_RATIO = 5

""""
param for RM:
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt'],
    'bootstrap': [True],
}

xgboost: "n_estimators": [200], "learning_rate": [0.01, 0.1]

"""
