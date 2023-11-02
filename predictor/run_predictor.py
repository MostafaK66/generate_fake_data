from utility import DataPreprocessor
import settings
from predictor.plotting import DataPlotter
import sys
import os
import pandas as pd
sys.path.append(os.path.abspath('../../'))
from generate_fake_data.mocked_up import run_ada



def main():
    run_ada.ada_df_generator()
    preprocessor = DataPreprocessor(split_ratio=settings.SPLIT_RATIO)
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

    return ada_projects, all_train_test_splits





if __name__ == "__main__":
    ada_projects, all_train_test_splits  = main()
    df1, df2, df3 = ada_projects
    splits_col1_df1, splits_col1_df2, splits_col1_df3 = all_train_test_splits[0]
    splits_col2_df1, splits_col2_df2, splits_col2_df3 = all_train_test_splits[1]


    print("yes")

