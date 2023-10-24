from utility import TimeSeriesPreprocessor
import settings


def main():
    preprocessor = TimeSeriesPreprocessor()
    df = preprocessor.read_data(filename=settings.filename)
    ada_project_1, ada_project_2, ada_project_3 = preprocessor.split_and_process(df=df)
    ada_project_1 = preprocessor.cumulative_done_per_pi(df=ada_project_1)
    ada_project_2 = preprocessor.cumulative_done_per_pi(df=ada_project_2)
    ada_project_3 = preprocessor.cumulative_done_per_pi(df=ada_project_3)

    return ada_project_1, ada_project_2, ada_project_3


if __name__ == "__main__":
    ada_project_1, ada_project_2, ada_project_3 = main()
