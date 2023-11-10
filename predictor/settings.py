from pathlib import Path

from scipy.stats import randint, uniform

script_dir = Path(__file__).resolve().parent
filename = (
    script_dir / ".." / "mocked_up" / "ada_output" / "ada_df_generator_output.csv"
)
N_IN = 4
N_OUT = 1
SPLIT_RATIO = 0.80
PARAM_DISTRIBUTION = {
    "xgb__n_estimators": randint(50, 100),
    "xgb__booster": ["gbtree", "dart"],
    "xgb__learning_rate": uniform(0.01, 0.10),
    "xgb__max_depth": randint(4, 6, 20),
}
TIME_SERIES_SPLIT_RATIO = 5
NUMBER_OF_RANDOMIZED_ITERATIONS = 3
