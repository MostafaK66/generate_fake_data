from utility import DataPreprocessor
import settings
from predictor.plotting import DataPlotter
import sys
import os

sys.path.append(os.path.abspath('../../'))
from generate_fake_data.mocked_up import run_ada



def main():
    run_ada.ada_df_generator()
    preprocessor = DataPreprocessor()
    plotter = DataPlotter()
    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_sort(df=df)
    ada_projects = [preprocessor.cumulative_done_per_date(df=project) for project in ada_projects]
    ada_projects = [preprocessor.cumulative_flow_per_date(df=project) for project in ada_projects]
    # ada_projects = [preprocessor.filter_columns(project) for project in ada_projects]
    # plotter.plot_projects(ada_projects)

    return ada_projects


if __name__ == "__main__":
    df_1, df_2, df_3 = main()
    print("yes")

