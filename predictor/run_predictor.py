from utility import DataPreprocessor
import settings
from plotting import DataPlotter


def main():
    preprocessor = DataPreprocessor()
    plotter = DataPlotter()
    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_process(df=df)
    ada_projects = [preprocessor.cumulative_done_per_pi(df=project) for project in ada_projects]
    ada_projects = [preprocessor.cumulative_flow_per_pi(df=project) for project in ada_projects]
    ada_projects = [preprocessor.filter_columns(project) for project in ada_projects]
    plotter.plot_projects(ada_projects)

    return ada_projects


if __name__ == "__main__":
    ada_project_1, ada_project_2, ada_project_3 = main()
    print("yes")

