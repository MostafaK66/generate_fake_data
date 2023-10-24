import pandas as pd
import os


class TimeSeriesPreprocessor:
    def __init__(self):

        pass

    def read_data(self, filename=None):

        if filename is None:

            script_dir = os.path.dirname(__file__)

            relative_path = "../mocked_up/ada_output/ada_df_generator_output.csv"

            filename = os.path.join(script_dir, relative_path)

        try:
            return pd.read_csv(filename, dtype={"PI": str})
        except FileNotFoundError:
            print(f"File not found: {filename}")
            return None
        except Exception as e:
            print(f"An error occurred while reading the file: {filename}, Error: {e}")
            return None
