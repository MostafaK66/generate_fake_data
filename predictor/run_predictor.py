import logging

logging.basicConfig(level=logging.WARNING)
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import settings
from sequence_predictor import UnivarientSequencePredictor
from tqdm import tqdm
from utility import DataPreprocessor

from predictor.plotting import DataPlotter

sys.path.append(os.path.abspath("../../"))
from generate_fake_data.mocked_up import run_ada

script_dir = Path(__file__).resolve().parent
params_dir = script_dir / "params"
params_dir.mkdir(exist_ok=True)
params_file_path = params_dir / "best_params_dict.txt"


def main():
    start_time = time.time()
    run_ada.ada_df_generator()
    preprocessor = DataPreprocessor(split_ratio=settings.SPLIT_RATIO)
    predictor = UnivarientSequencePredictor(
        param_distributions=settings.PARAM_DISTRIBUTION,
        time_series_split_ratio=settings.TIME_SERIES_SPLIT_RATIO,
        n_iter=settings.NUMBER_OF_RANDOMIZED_ITERATIONS,
    )
    plotter = DataPlotter()
    walk_forward_validation_results = {}
    col_names = ["PredictedDone", "PredictedFlow"]
    best_params_dict = {}
    mae_dict = {}

    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_sort(df=df)
    ada_projects = [
        preprocessor.done_tickets_per_date(df=project) for project in ada_projects
    ]
    ada_projects = [preprocessor.flow_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.filter_dataframe(project) for project in ada_projects]
    ada_projects = [
        preprocessor.fill_consecutive_dates(project) for project in ada_projects
    ]
    plotter.plot_projects(
        ada_projects=ada_projects, split_ratio=settings.SPLIT_RATIO, last_n_days=None
    )

    all_transformed_dfs = []
    for col_idx in tqdm(
        [0, 1], desc="Transforming columns to seupervised learning sequence"
    ):
        transformed_dfs = preprocessor.process_dataframe_series(
            ada_projects, col_idx, n_in=settings.N_IN, n_out=settings.N_OUT
        )
        all_transformed_dfs.append(transformed_dfs)

    all_train_test_splits = []
    for transformed_dfs in tqdm(
        all_transformed_dfs, desc="Splitting train and test data"
    ):
        train_test_splits = [
            preprocessor.train_test_split(data) for data in transformed_dfs
        ]
        all_train_test_splits.append(train_test_splits)

    total_iterations = sum(len(splits) for splits in all_train_test_splits)
    progress_bar = tqdm(total=total_iterations, desc="Running walk forward validation")
    mae_sum = 0
    for col_idx, train_test_splits in enumerate(all_train_test_splits):
        for df_idx, split in enumerate(train_test_splits):
            train, test = split
            mae, results_df, best_params = predictor.walk_forward_validation(
                train, test
            )
            key = f"splits_{col_names[col_idx]}_df{df_idx + 1}"
            walk_forward_validation_results[key] = {"MAE": mae, "ResultsDF": results_df}
            best_params_dict[key] = best_params
            mae_dict[key] = mae
            mae_sum += mae

            progress_bar.update(1)
    with open(params_file_path, "w") as file:
        file.write(json.dumps(best_params_dict, indent=4))

    progress_bar.close()
    average_mae = mae_sum / total_iterations

    plotter.process_and_plot(
        ada_projects=ada_projects,
        walk_forward_validation_results=walk_forward_validation_results,
        split_ratio=settings.SPLIT_RATIO,
        n_in=settings.N_IN,
        last_n_days=None,
    )
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time ML pipeline: {total_time:.2f} seconds.")

    return (
        ada_projects,
        all_train_test_splits,
        walk_forward_validation_results,
        best_params_dict,
        mae_dict,
        average_mae,
    )


if __name__ == "__main__":
    (
        ada_projects,
        all_train_test_splits,
        walk_forward_validation_results,
        best_params_dict,
        mae_dict,
        average_mae,
    ) = main()
    df1, df2, df3 = ada_projects
    (
        splits_PredictedDone_df1,
        splits_PredictedDone_df2,
        splits_PredictedDone_df3,
    ) = all_train_test_splits[0]
    (
        splits_PredictedFlow_df1,
        splits_PredictedFlow_df2,
        splits_PredictedFlow_df3,
    ) = all_train_test_splits[1]
    result_splits_PredictedDone_df1 = walk_forward_validation_results[
        "splits_PredictedDone_df1"
    ]
    result_splits_PredictedDone_df2 = walk_forward_validation_results[
        "splits_PredictedDone_df2"
    ]
    result_splits_PredictedDone_df3 = walk_forward_validation_results[
        "splits_PredictedDone_df3"
    ]
    result_splits_PredictedFlow_df1 = walk_forward_validation_results[
        "splits_PredictedFlow_df1"
    ]
    result_splits_PredictedFlow_df2 = walk_forward_validation_results[
        "splits_PredictedFlow_df2"
    ]
    result_splits_PredictedFlow_df3 = walk_forward_validation_results[
        "splits_PredictedFlow_df3"
    ]

    print(f"best parameters: {best_params_dict}")
    print(f"calculated average MAE: {average_mae}")
