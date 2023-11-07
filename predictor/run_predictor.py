from utility import DataPreprocessor
import settings
from predictor.plotting import DataPlotter
from sequence_predictor import UnivarientSequencePredictor
import sys
import os
import pandas as pd

sys.path.append(os.path.abspath('../../'))
from generate_fake_data.mocked_up import run_ada

def main():
    run_ada.ada_df_generator()
    preprocessor = DataPreprocessor(split_ratio=settings.SPLIT_RATIO)
    predictor = UnivarientSequencePredictor()
    plotter = DataPlotter()

    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_sort(df=df)
    ada_projects = [preprocessor.done_tickets_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.flow_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.filter_dataframe(project) for project in ada_projects]
    ada_projects = [preprocessor.fill_consecutive_dates(project) for project in ada_projects]
    plotter.plot_projects(ada_projects=ada_projects, last_n_days=None)

    all_transformed_dfs = []
    for col_idx in [0, 1]:
        transformed_dfs = preprocessor.process_dataframe_series(ada_projects, col_idx, n_in=settings.N_IN,
                                                                n_out=settings.N_OUT)
        all_transformed_dfs.append(transformed_dfs)

    all_train_test_splits = []
    for transformed_dfs in all_transformed_dfs:
        train_test_splits = [preprocessor.train_test_split(data) for data in transformed_dfs]
        all_train_test_splits.append(train_test_splits)

    walk_forward_validation_results = {}
    col_names = ["PredictedDone", "PredictedFlow"]
    for col_idx, train_test_splits in enumerate(all_train_test_splits):
        for df_idx, split in enumerate(train_test_splits):
            train, test = split
            result = predictor.walk_forward_validation(train, test)
            key = f"splits_{col_names[col_idx]}_df{df_idx + 1}"
            walk_forward_validation_results[key] = result

    return ada_projects, all_train_test_splits, walk_forward_validation_results

if __name__ == "__main__":
    ada_projects, all_train_test_splits, walk_forward_validation_results = main()
    df1, df2, df3 = ada_projects
    splits_col1_df1, splits_col1_df2, splits_col1_df3 = all_train_test_splits[0]
    splits_col2_df1, splits_col2_df2, splits_col2_df3 = all_train_test_splits[1]
    a, b = splits_col1_df1
    result_splits_PredictedDone_df1 = walk_forward_validation_results['splits_PredictedDone_df1']
    result_splits_PredictedDone_df2 = walk_forward_validation_results['splits_PredictedDone_df2']
    result_splits_PredictedDone_df3 = walk_forward_validation_results['splits_PredictedDone_df3']
    result_splits_PredictedFlow_df1 = walk_forward_validation_results['splits_PredictedFlow_df1']
    result_splits_PredictedFlow_df2 = walk_forward_validation_results['splits_PredictedFlow_df2']
    result_splits_PredictedFlow_df3 = walk_forward_validation_results['splits_PredictedFlow_df3']

    print("Results for splits_col1_df1:", result_splits_col1_df1)






