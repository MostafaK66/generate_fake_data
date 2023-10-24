from utility import TimeSeriesPreprocessor
import settings


def main():
    preprocessor = TimeSeriesPreprocessor()
    df = preprocessor.read_data(filename=settings.filename)

    return df


if __name__ == "__main__":
    df = main()
