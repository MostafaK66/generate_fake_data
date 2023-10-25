from utility import TimeSeriesPreprocessor
import settings


def main():
    preprocessor = TimeSeriesPreprocessor()
    df = preprocessor.read_data(filename=settings.filename)
    ada_projects = preprocessor.split_and_process(df=df)
    ada_projects = [preprocessor.cumulative_done_per_pi(df=project) for project in ada_projects]
    ada_projects = [preprocessor.cumulative_flow_per_pi(df=project) for project in ada_projects]

    return ada_projects


if __name__ == "__main__":
    ada_project_1, ada_project_2, ada_project_3 = main()
    print("yes")

