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
    preprocessor = DataPreprocessor()
    plotter = DataPlotter()
    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_sort(df=df)
    ada_projects = [preprocessor.done_tickets_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.flow_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.filter_dataframe(project) for project in ada_projects]
    ada_projects = [preprocessor.fill_consecutive_dates(project) for project in ada_projects]
    plotter.plot_projects(ada_projects=ada_projects, last_n_days=21)

    all_transformed_dfs = []

    for col_idx in [0, 1]:
        transformed_dfs = preprocessor.process_dataframe_series(ada_projects, col_idx, n_in=settings.N_IN,
                                                                n_out=settings.N_OUT)
        all_transformed_dfs.append(transformed_dfs)

    return ada_projects, all_transformed_dfs




if __name__ == "__main__":
    ada_projects, all_transformed_dfs = main()
    df1, df2, df3 = ada_projects
    ar1_col1, ar2_col1, ar3_col1 = all_transformed_dfs[0]
    ar1_col2, ar2_col2, ar3_col2 = all_transformed_dfs[1]
    print("yes")

