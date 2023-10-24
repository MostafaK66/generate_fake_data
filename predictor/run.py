from utility import TimeSeriesPreprocessor


def main():
    preprocessor = TimeSeriesPreprocessor()
    df = preprocessor.read_data()

    return df


if __name__ == "__main__":
    df = main()
