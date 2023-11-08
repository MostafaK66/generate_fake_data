from pathlib import Path

script_dir = Path(__file__).resolve().parent
filename = (
    script_dir / ".." / "mocked_up" / "ada_output" / "ada_df_generator_output.csv"
)
N_IN = 4
N_OUT = 1
SPLIT_RATIO = 0.80
RF_PARAM_GRID = {"n_estimators": [1]}
TIME_SERIES_SPLIT_RATIO = 3
